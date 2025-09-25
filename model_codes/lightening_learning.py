from sklearn.metrics import r2_score
from Models3 import *
import torch.nn.functional as F
from Lightening_dataset import *
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger("tb_logs", name="my_model_experiment")
import pytorch_lightning as pl

import torch.nn.functional as F

def mse_error(x, x_head, mask):
    # Apply the mask to both ground truth and prediction
    x_masked = x * mask
    x_head_masked = x_head * mask

    # Compute the maximum value over the last two dimensions
    # Using torch.amax is a cleaner way to do this than sequential max() calls
    x_max = torch.amax(x_masked, dim=(-2, -1))
    x_head_max = torch.amax(x_head_masked, dim=(-2, -1))

    # Compute and return the MSE loss
    loss = F.mse_loss(x_max, x_head_max)

    # Get the overall max values of the masked tensors for a different return value
    # You were using .max() on the original tensors, which is inconsistent
    x_max_all7 = x_masked.max()
    x_head_max_all7 = x_head_masked.max()

    return loss, x_max_all7.item(), x_head_max_all7.item()

OBJECTS = np.load('Data4\\objects_random2.npy')
MASKS = np.load('Data4\\masks_random2.npy')
OPTICSES = np.load('Data4\\optics_random2.npy')
OPTICSES = np.concatenate([OPTICSES[:,:2], OPTICSES[:,:2]], axis = 1)

N_Sample = OPTICSES.shape[0]
#%%
def Generate_Reconstruction_Image_Fast(n = N_Sample):
    sample_indxes = np.random.randint(0, n, size=16)
    return OBJECTS[sample_indxes],  MASKS[sample_indxes], OPTICSES[sample_indxes]


def Weighted_Loss(x, x_head, mask):
    # Compute the maximum value over the last two dimensions sequentially
    x_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    # Normalize the input tensors by the maximum value
    x_normalized = x / (x_max + 1e-6) * mask
    x_head_normalized = x_head / (x_max + 1e-6) * mask

    # mask_ind = mask<0.01

    # Compute and return the L1 loss
    return F.l1_loss(x, x_head) + F.l1_loss(x_normalized,
                                            x_head_normalized)  # + F.l1_loss(x_head[mask_ind],mask[mask_ind])


class YourLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Disable automatic optimization
        self.automatic_optimization = False
        self.inverse_model = Inverse_Operator()
        self.forward_model = Forward_Operator()
        self.dis = Discriminator(input_shape=7520)
        self.bce = nn.BCEWithLogitsLoss()

        # Lists for storing step outputs for compatibility with older PL versions
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, mask, pert_label, b_optics):
        return self.inverse_model(mask, pert_label, b_optics)

    def training_step(self, batch, batch_idx):
        opt_inverse, opt_forward, opt_dis = self.optimizers()

        pert_label, mask, b_optics, _, Mua_Ground_Truth = batch

        # ------------------------------------
        # Training Loop 1: Train Inverse and Forward models (P2P)
        # ------------------------------------
        opt_inverse.zero_grad()
        opt_forward.zero_grad()
        Mua_Recons = self.inverse_model(mask, pert_label, b_optics)
        pert_pred = self.forward_model(Mua_Recons, b_optics)
        loss_pert = F.l1_loss(pert_label[:, :18, :], pert_pred)
        # Use a correctly expanded mask for the loss calculation
        mask_expanded = mask.unsqueeze(1).expand_as(Mua_Ground_Truth)
        loss_recons1 = Weighted_Loss(Mua_Ground_Truth * mask_expanded, Mua_Recons, mask_expanded) * 1.5
        loss_recons = loss_recons1 + loss_pert
        self.manual_backward(loss_recons)
        opt_inverse.step()
        opt_forward.step()
        self.log('P2P/loss_recons', loss_recons1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('P2P/loss_pert', loss_pert, on_step=True, on_epoch=True, prog_bar=True)

        # ------------------------------------
        # Training Loop 2: Train Discriminator
        # ------------------------------------
        opt_dis.zero_grad()
        D_real = self.dis(mask, pert_label, torch.cat(
            [b_optics[:, :2].repeat([1, 20]), Mua_Ground_Truth.amax(dim=[1, 2, 3]).unsqueeze(1).repeat([1, 60])],
            axis=1))
        D_real_loss = self.bce(D_real, torch.ones_like(D_real))
        pert_pred_noisy = pert_pred.detach() + torch.randn_like(pert_pred.detach()) * 0.25
        D_fake = self.dis(mask, pert_pred_noisy, torch.cat(
            [b_optics[:, :2].repeat([1, 20]), Mua_Ground_Truth.amax(dim=[1, 2, 3]).unsqueeze(1).repeat([1, 60])],
            axis=1))
        D_fake_loss = self.bce(D_fake, torch.zeros_like(D_fake))
        Mua_GT_generated, mask_gen, optics_gen = Generate_Reconstruction_Image_Fast()
        Mua_GT_generated = Mua_GT_generated.to(self.device)
        mask_gen = mask_gen.to(self.device)
        optics_gen = optics_gen.to(self.device)
        pert_pred2 = self.forward_model(Mua_GT_generated, optics_gen)
        pert_pred2_noisy = pert_pred2.detach() + torch.randn_like(pert_pred2.detach()) * 0.25
        D_fake2 = self.dis(mask_gen, pert_pred2_noisy, torch.cat(
            [optics_gen[:, :2].repeat([1, 20]), Mua_GT_generated.amax(dim=[1, 2, 3]).unsqueeze(1).repeat([1, 60])],
            axis=1))
        D_fake_loss2 = self.bce(D_fake2, torch.zeros_like(D_fake2))
        D_loss = (D_fake_loss * 0.2 + D_real_loss + D_fake_loss2) / 3
        self.manual_backward(D_loss)
        opt_dis.step()
        self.log('D_loss/total', D_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('D_loss/D_fake', D_fake_loss, on_step=True)
        self.log('D_loss/D_fake2', D_fake_loss2, on_step=True)
        self.log('D_loss/D_real', D_real_loss, on_step=True)

        # ------------------------------------
        # Training Loop 3: Train Inverse and Forward models (R2R)
        # ------------------------------------
        opt_inverse.zero_grad()
        opt_forward.zero_grad()
        Mua_Recons_r2r = self.inverse_model(mask_gen, pert_pred2, optics_gen)
        D_fake3 = self.dis(mask_gen, pert_pred2, torch.cat(
            [optics_gen[:, :2].repeat(1, 20), Mua_GT_generated.amax(dim=[1, 2, 3]).unsqueeze(1).repeat([1, 60])],
            axis=1))
        G_fake_loss = self.bce(D_fake3, torch.ones_like(D_fake3))
        mask_gen_expanded = mask_gen.unsqueeze(1).expand_as(Mua_GT_generated)
        loss_recons3 = Weighted_Loss(Mua_GT_generated * mask_gen_expanded, Mua_Recons_r2r, mask_gen_expanded) * 1.5
        loss_recons5 = loss_recons3 + G_fake_loss * 0.001
        self.manual_backward(loss_recons5)
        opt_inverse.step()
        opt_forward.step()
        self.log('R2R/loss_recons', loss_recons3, on_step=True, on_epoch=True)
        self.log('R2R/G_fake_loss', G_fake_loss, on_step=True, on_epoch=True)

        # Append outputs for on_training_epoch_end
        self.training_step_outputs.append({'loss_recons': loss_recons, 'D_loss': D_loss, 'loss_recons5': loss_recons5})

    def on_training_epoch_end(self):
        # Optional: Log average training metrics at the end of the epoch
        avg_loss_recons = torch.stack([x['loss_recons'] for x in self.training_step_outputs]).mean()
        avg_d_loss = torch.stack([x['D_loss'] for x in self.training_step_outputs]).mean()
        avg_loss_recons5 = torch.stack([x['loss_recons5'] for x in self.training_step_outputs]).mean()

        self.log('epoch/avg_train_recons_loss', avg_loss_recons, on_epoch=True)
        self.log('epoch/avg_train_d_loss', avg_d_loss, on_epoch=True)
        self.log('epoch/avg_train_r2r_loss', avg_loss_recons5, on_epoch=True)

        # Clear the list after the epoch ends
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        opt_inverse = optim.Adam(self.inverse_model.parameters(), lr=0.00002, betas=(0.5, 0.999))
        opt_forward = optim.Adam(self.forward_model.parameters(), lr=0.00002, betas=(0.5, 0.999))
        opt_dis = optim.Adam(self.dis.parameters(), lr=0.00002, betas=(0.5, 0.999))
        return (
            {'optimizer': opt_inverse, 'loss': 'loss_recons', 'frequency': 1},
            {'optimizer': opt_forward, 'loss': 'loss_recons', 'frequency': 1},
            {'optimizer': opt_dis, 'loss': 'D_loss', 'frequency': 1}
        )

    def validation_step(self, batch, batch_idx):
        pert_label, mask, b_optics, _, Mua_Ground_Truth = batch
        Mua_Recons = self.inverse_model(mask, pert_label, b_optics)
        print()
        # Expand mask to match Mua_Ground_Truth shape
        mask_expanded = mask.unsqueeze(1).expand_as(Mua_Ground_Truth)
        mse, gt, pred = mse_error(Mua_Ground_Truth, Mua_Recons, mask_expanded)

        self.log('val_mse', mse, on_step=False, on_epoch=True, prog_bar=True)

        # Append outputs for on_validation_epoch_end
        output = {'gt': gt, 'pred': pred, 'Mua_Ground_Truth': Mua_Ground_Truth, 'Mua_Recons': Mua_Recons, 'mask': mask}
        self.validation_step_outputs.append(output)

        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        # Check if outputs list is empty to prevent errors during sanity check
        if not outputs:
            return

        gt_max = torch.cat([x['gt'] for x in outputs])
        pred_max = torch.cat([x['pred'] for x in outputs])
        r2 = r2_score(gt_max.cpu().numpy(), pred_max.cpu().numpy())
        self.log('val_r2_score', r2, on_epoch=True, prog_bar=True)

        # Plotting the first 5 images
        if self.current_epoch % 3 == 0:
            first_batch_outputs = outputs[0]
            Mua_Ground_Truth_val = first_batch_outputs['Mua_Ground_Truth']
            Mua_Recons_val = first_batch_outputs['Mua_Recons']
            mask_val = first_batch_outputs['mask']

            for i in range(min(5, Mua_Ground_Truth_val.shape[0])):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                ax = axes[0]
                img_gt = show_tensor_image2(Mua_Ground_Truth_val[i])
                ax.imshow(img_gt, cmap='viridis')
                ax.set_title('Ground Truth')

                ax = axes[1]
                img_recons = show_tensor_image2(Mua_Recons_val[i])
                ax.imshow(img_recons, cmap='viridis')
                ax.set_title('Reconstructed Image')

                ax = axes[2]
                img_mask = show_tensor_image2(mask_val[i])
                ax.imshow(img_mask, cmap='gray')
                ax.set_title('Mask')

                plt.suptitle(f'Epoch {self.current_epoch} - Validation Image {i + 1}')

                # Add the figure to TensorBoard
                self.logger.experiment.add_figure(f"Validation_Epoch_{self.current_epoch}/Image_{i + 1}", fig,
                                                  self.current_epoch)
                plt.close(fig)

        # Clear the list after the epoch ends
        self.validation_step_outputs.clear()

if __name__ == '__main__':

    BATCH_SIZE = 6

    train_dataset = MyDataset(
        X_train, y_train, Mask_train, optics_train, target_train, train=True
    )

    test_dataset = MyDataset(
        X_test, y_test, Mask_test, optics_test, target_test, train=False
    )

    # Create the training DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate the data module and model
    model = YourLightningModel()
    trainer = pl.Trainer(
        max_epochs=100,
        logger=logger,  # Pass the logger here
        accelerator="auto"  # use "gpu" if you want to specify a device
    )

    trainer.fit(model, train_dataloader, test_dataloader)