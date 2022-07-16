import os

import torch
from tqdm import tqdm

from utils.utils import get_lr, show_result


def fit_one_epoch(G_model_A2B_train, G_model_B2A_train, D_model_A_train, D_model_B_train, G_model_A2B, G_model_B2A, D_model_A, D_model_B, loss_history, 
                G_optimizer, D_optimizer_A, D_optimizer_B, BCE_loss, MSE_loss, epoch, epoch_step, gen, Epoch, cuda, fp16, scaler, save_period, save_dir, photo_save_step, local_rank=0):
    G_total_loss    = 0
    D_total_loss_A  = 0
    D_total_loss_B  = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        
        images_A, images_B = batch[0], batch[1]
        batch_size  = images_A.size()[0]
        y_real      = torch.ones(batch_size)
        y_fake      = torch.zeros(batch_size)
        
        with torch.no_grad():
            if cuda:
                images_A, images_B, y_real, y_fake  = images_A.cuda(local_rank), images_B.cuda(local_rank), y_real.cuda(local_rank), y_fake.cuda(local_rank)

        if not fp16:
            #---------------------------------#
            #   训练生成器A2B和B2A
            #---------------------------------#
            G_optimizer.zero_grad()
            
            Same_B          = G_model_A2B_train(images_B)
            loss_identity_B = MSE_loss(Same_B, images_B)
            
            Same_A          = G_model_B2A_train(images_A)
            loss_identity_A = MSE_loss(Same_A, images_A)
            
            fake_B          = G_model_A2B_train(images_A)
            pred_fake       = D_model_B_train(fake_B)
            loss_GAN_A2B    = BCE_loss(pred_fake, y_real)

            fake_A          = G_model_B2A_train(images_B)
            pred_fake       = D_model_A_train(fake_A)
            loss_GAN_B2A    = BCE_loss(pred_fake, y_real)
            
            recovered_A     = G_model_B2A_train(fake_B)
            loss_cycle_ABA  = MSE_loss(recovered_A, images_A)

            recovered_B     = G_model_A2B_train(fake_A)
            loss_cycle_BAB  = MSE_loss(recovered_B, images_B)

            G_loss = loss_identity_A * 5.0 + loss_identity_B * 5.0 + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA * 10.0 + loss_cycle_BAB * 10.0
            G_loss.backward()
            G_optimizer.step()
                
            #---------------------------------#
            #   训练评价器A
            #---------------------------------#
            D_optimizer_A.zero_grad()
            pred_real   = D_model_A_train(images_A)
            loss_D_real = BCE_loss(pred_real, y_real)

            pred_fake   = D_model_A_train(fake_A.detach())
            loss_D_fake = BCE_loss(pred_fake, y_fake)

            D_loss_A    = (loss_D_real + loss_D_fake) * 0.5
            D_loss_A.backward()
            D_optimizer_A.step()
            
            #---------------------------------#
            #   训练评价器B
            #---------------------------------#
            D_optimizer_B.zero_grad()

            pred_real   = D_model_B_train(images_B)
            loss_D_real = BCE_loss(pred_real, y_real)

            pred_fake   = D_model_B_train(fake_B.detach())
            loss_D_fake = BCE_loss(pred_fake, y_fake)

            D_loss_B = (loss_D_real + loss_D_fake) * 0.5
            D_loss_B.backward()
            D_optimizer_B.step()

        else:
            from torch.cuda.amp import autocast

            #---------------------------------#
            #   训练生成器A2B和B2A
            #---------------------------------#
            with autocast():
                G_optimizer.zero_grad()
                Same_B          = G_model_A2B_train(images_B)
                loss_identity_B = MSE_loss(Same_B, images_B)
                
                Same_A          = G_model_B2A_train(images_A)
                loss_identity_A = MSE_loss(Same_A, images_A)
                
                fake_B          = G_model_A2B_train(images_A)
                pred_fake       = D_model_B_train(fake_B)
                
                loss_GAN_A2B    = BCE_loss(pred_fake, y_real)

                fake_A          = G_model_B2A_train(images_B)
                pred_fake       = D_model_A_train(fake_A)
                loss_GAN_B2A    = BCE_loss(pred_fake, y_real)
                
                recovered_A     = G_model_B2A_train(fake_B)
                loss_cycle_ABA  = MSE_loss(recovered_A, images_A)

                recovered_B     = G_model_A2B_train(fake_A)
                loss_cycle_BAB  = MSE_loss(recovered_B, images_B)

                G_loss = loss_identity_A * 5.0 + loss_identity_B * 5.0 + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA * 10.0 + loss_cycle_BAB * 10.0
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(G_loss).backward()
            scaler.step(G_optimizer)
            scaler.update()
            
            #---------------------------------#
            #   训练评价器A
            #---------------------------------#
            with autocast():
                D_optimizer_A.zero_grad()
                pred_real   = D_model_A_train(images_A)
                loss_D_real = BCE_loss(pred_real, y_real)

                pred_fake   = D_model_A_train(fake_A.detach())
                loss_D_fake = BCE_loss(pred_fake, y_fake)

                D_loss_A = (loss_D_real + loss_D_fake) * 0.5
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(D_loss_A).backward()
            scaler.step(D_optimizer_A)
            scaler.update()
            
            #---------------------------------#
            #   训练评价器B
            #---------------------------------#
            with autocast():
                D_optimizer_B.zero_grad()

                pred_real   = D_model_B_train(images_B)
                loss_D_real = BCE_loss(pred_real, y_real)

                pred_fake   = D_model_B_train(fake_B.detach())
                loss_D_fake = BCE_loss(pred_fake, y_fake)

                D_loss_B = (loss_D_real + loss_D_fake) * 0.5
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(D_loss_B).backward()
            scaler.step(D_optimizer_B)
            scaler.update()
                
        G_total_loss    += G_loss.item()
        D_total_loss_A  += D_loss_A.item()
        D_total_loss_B  += D_loss_B.item()

        if local_rank == 0:
            pbar.set_postfix(**{'G_loss'    : G_total_loss / (iteration + 1), 
                                'D_loss_A'  : D_total_loss_A / (iteration + 1), 
                                'D_loss_B'  : D_total_loss_B / (iteration + 1), 
                                'lr'        : get_lr(G_optimizer)})
            pbar.update(1)

            if iteration % photo_save_step == 0:
                show_result(epoch + 1, G_model_A2B, G_model_B2A, images_A, images_B)

    G_total_loss    = G_total_loss / epoch_step
    D_total_loss_A  = D_total_loss_A / epoch_step
    D_total_loss_B  = D_total_loss_B / epoch_step
    
    if local_rank == 0:
        pbar.close()
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('G Loss: %.4f || D Loss A: %.4f || D Loss B: %.4f  ' % (G_total_loss, D_total_loss_A, D_total_loss_B))
        loss_history.append_loss(epoch + 1, G_total_loss = G_total_loss, D_total_loss_A = D_total_loss_A, D_total_loss_B = D_total_loss_B)

        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(G_model_A2B.state_dict(), os.path.join(save_dir, 'G_model_A2B_Epoch%d-GLoss%.4f-DALoss%.4f-DBLoss%.4f.pth'%(epoch + 1, G_total_loss, D_total_loss_A, D_total_loss_B)))
            torch.save(G_model_B2A.state_dict(), os.path.join(save_dir, 'G_model_B2A_Epoch%d-GLoss%.4f-DALoss%.4f-DBLoss%.4f.pth'%(epoch + 1, G_total_loss, D_total_loss_A, D_total_loss_B)))
            torch.save(D_model_A.state_dict(), os.path.join(save_dir, 'D_model_A_Epoch%d-GLoss%.4f-DALoss%.4f-DBLoss%.4f.pth'%(epoch + 1, G_total_loss, D_total_loss_A, D_total_loss_B)))
            torch.save(D_model_B.state_dict(), os.path.join(save_dir, 'D_model_B_Epoch%d-GLoss%.4f-DALoss%.4f-DBLoss%.4f.pth'%(epoch + 1, G_total_loss, D_total_loss_A, D_total_loss_B)))

        torch.save(G_model_A2B.state_dict(), os.path.join(save_dir, "G_model_A2B_last_epoch_weights.pth"))
        torch.save(G_model_B2A.state_dict(), os.path.join(save_dir, "G_model_B2A_last_epoch_weights.pth"))
        torch.save(D_model_A.state_dict(), os.path.join(save_dir, "D_model_A_last_epoch_weights.pth"))
        torch.save(D_model_B.state_dict(), os.path.join(save_dir, "D_model_B_last_epoch_weights.pth"))