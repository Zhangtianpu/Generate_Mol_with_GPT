import torch

import Utils
import time
import os
import numpy as np


def train_model_smiple(model,
                       train_loader,
                       val_loader,
                       optimizer, device,
                       num_epochs, eval_freq, eval_iter,
                       start_context,
                       tokenizer,
                       save_model_path):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        best_loss = 10000
        start_time = time.time()
        for input, target in train_loader:
            optimizer.zero_grad()
            batch_loss = Utils.calc_loss_batch(input, target, model, device)
            batch_loss.backward()
            optimizer.step()
            tokens_seen += input.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                end_time = time.time()
                training_time = (end_time - start_time) / 60
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print('epoch:%d\tglobal step:%09d\ttrain loss:%.3f\tval loss:%.3f\ttraining time:%.3fmin' % (
                    epoch + 1, global_step, train_loss, val_loss, training_time))

                if device == 0 and best_loss > val_loss:
                    best_loss = val_loss
                    Utils.save_model(model, save_model_path)

        generate_and_print_simple(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def train_GAN_model(molgan, base_config,discriminator,
                    train_loader,
                    val_loader,
                    G_optimizer, D_optimizer, loss_fn, device,
                    num_epochs, eval_freq, eval_iter,
                    start_context,
                    tokenizer,
                    save_model_folder):
    train_D_losses, train_G_losses, track_tokens_seen = [], [], []
    val_D_losses, val_G_losses = [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        molgan.train()
        discriminator.train()
        best_loss = 10000
        start_time = time.time()
        for index, data in enumerate(train_loader,start=1):

            input, target = data
            input, target = input.to(device), target.to(device)
            b, t = input.shape
            """
            sample fake data from given distribution
            """
            fake_input=torch.randn(size=(b,t,base_config['emb_dim']),dtype=torch.float32).to(device)
            """
            Train discriminator
            """
            real_input = molgan.pre_trained_llm(input)
            fake_input = molgan(fake_input)
            D_optimizer.zero_grad()

            real_labels = torch.ones(b, t, 1).to(device)
            fake_labels = torch.zeros(b,t, 1).to(device)

            real_out = discriminator(real_input)
            fake_out = discriminator(fake_input.detach())
            D_loss_real = loss_fn(real_out, real_labels)
            D_loss_fake = loss_fn(fake_out, fake_labels)
            D_loss = D_loss_fake + D_loss_real
            D_loss.backward()
            D_optimizer.step()

            """
            Train generator
            """
            if index % 5 == 0:
                G_optimizer.zero_grad()
                fake_out = discriminator(fake_input)
                G_loss = loss_fn(fake_out, real_labels)
                G_loss.backward()
                G_optimizer.step()

            tokens_seen += input.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                end_time = time.time()
                training_time = (end_time - start_time) / 60
                train_D_loss, train_G_loss = evaluate_GAN_model(molgan,base_config, discriminator, loss_fn, train_loader, device,
                                                                eval_iter)
                val_D_loss, val_G_loss = evaluate_GAN_model(molgan,base_config, discriminator, loss_fn, val_loader, device,
                                                            eval_iter)

                train_D_losses.append(train_D_loss)
                train_G_losses.append(train_G_loss)
                val_D_losses.append(val_D_loss)
                val_G_losses.append(val_G_loss)

                track_tokens_seen.append(tokens_seen)
                print('epoch:%d\tglobal step:%09d\ttrain D loss:%.3f\ttrain G loss:%.3f\ttraining time:%.3fmin' % (
                    epoch + 1, global_step, train_D_loss, train_G_loss, training_time))
                print('epoch:%d\tglobal step:%09d\tval D loss:%.3f\tval G loss:%.3f' % (
                    epoch + 1, global_step, val_D_loss, val_G_loss))
                GAN_loss = val_D_loss + val_G_loss
                if device == 0 and best_loss > GAN_loss:
                    best_loss = GAN_loss

                    Utils.save_model(molgan, os.path.join(save_model_folder, 'molgan.pkl'))
                    Utils.save_model(discriminator, os.path.join(save_model_folder, 'discriminator.pkl'))

        generate_and_print_simple_based_GAN(molgan,base_config,tokenizer)
    return train_D_losses, train_G_losses, val_D_losses, val_G_losses, track_tokens_seen


def evaluate_GAN_model(molgan,base_config, discriminator, loss_fn, data_loader, device, eval_iter):
    with torch.no_grad():
        total_loss_D = 0
        total_loss_G = 0
        if eval_iter is None:
            num_batch = len(data_loader)
        else:
            num_batch = min(eval_iter, len(data_loader))

        for i, input in enumerate(data_loader):
            input = input[0].to(device)
            if i < num_batch:
                b,t=input.shape

                """
                sample fake data from given distribution
                """

                fake_input = torch.rand(size=(b, t, base_config['emb_dim']), dtype=torch.float32).to(device)
                real_labels = torch.ones(b,t, 1).to(device)
                fake_labels = torch.zeros(b,t, 1).to(device)
                real_input = molgan.pre_trained_llm(input)
                fake_input = molgan(fake_input)
                real_out = discriminator(real_input)
                fake_out = discriminator(fake_input)
                real_loss = loss_fn(real_out, real_labels)
                fake_loss = loss_fn(fake_out, fake_labels)
                D_loss = real_loss + fake_loss

                G_loss = loss_fn(fake_out, real_labels)

                total_loss_D += D_loss
                total_loss_G += G_loss

            else:
                break
        return total_loss_D / num_batch, total_loss_G / num_batch


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = Utils.calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = Utils.calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_simple(model, tokenizer, device, start_context):
    model.eval()
    idx = Utils.text_to_ids(start_context, tokenizer).to(device)
    # context_size=model.token_layer.wordEmbeddingLayer.wordEmbed_layer.weight.shape[0]
    output_idx = Utils.generate(model, idx, max_new_tokens=100, context_size=256, top_K=30, temperature=1.5)
    output_text = Utils.ids_to_text(output_idx, tokenizer)
    print(output_text)
    # print(output_text.replace('\n', ' '))

    model.train()


def generate_and_print_simple_based_GAN(MolGAN,base_config, tokenizer):
    MolGAN.eval()
    output_idx = Utils.generate_based_GAN_random_noise(MolGAN, base_config, max_new_tokens=100)
    output_text = Utils.ids_to_text(output_idx, tokenizer)
    print(output_text)

    MolGAN.train()
