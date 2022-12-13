step = 0

for epoch in range(NUM_EPOCHS):
    #we will track the total loss of the generator and critic for each epoch over the entire dataset
    #initialize the total loss of the generator and critic for each epoch to 0
    total_loss_gen = 0
    total_loss_critic = 0
    
    #move these to device

    #have no gradient for these losse
    # total_loss_gen = torch.no_grad()
    # total_loss_critic = torch.no_grad()
    
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(loader):
        batch_step = 0
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            
            
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            
        #trained critic
        

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        
        
        
        
        #update the total loss of the generator and critic for each batch in the epoch
        #add just the value no gradients
        #have no gradient for these losse
        #add just the value no gradientsfrom the loss_gen tensor
      
        with torch.no_grad():
            total_loss_gen += loss_gen.item()
            total_loss_critic += loss_critic.item()
        
        # Print losses occasionally and print to tensorboard in a batch
        if batch_idx % 10 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
            
            with torch.no_grad():
                
                #AVERAGE LOSS---
        
                #get average loss of generator and critic for each epoch
                avg_loss_gen = total_loss_gen / len(loader)
                avg_loss_critic = total_loss_critic / len(loader)
                #write loss to tensorboard
                writer_loss.add_scalar("Generator loss Epoch", avg_loss_gen, global_step=batch_step)
                writer_loss.add_scalar("Critic loss Epoch", avg_loss_critic, global_step=batch_step)
                
                #AVERAGE LOSS----
                
                #FID--
                #calculate FID score of this batch
                #update the fid_score with real and fake images
                fid_score.update(interpolate(real), interpolate(fake))
                computed_fid_score = fid_score.compute()
                print("FID score: ", computed_fid_score)
                writer_loss.add_scalar("FID Score", computed_fid_score, global_step=batch_step)
                #reset the fid score
                fid_score.reset()
                ##FID--
                
                batch_step += 1
            
            
        
        

        # Print losses occasionally and print to tensorboard per epoch
        # if batch_idx % 100 == 0 and batch_idx > 0:
        
            ## PRINT for few epochs %10
            # print(
            #     f"Epoch [{epoch}/{NUM_EPOCHS}]  \
            #         Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            # )
            
    #calculate the Fréchet Inception Distance (FID) to evaluate the performance of the generator
    

    with torch.no_grad():
        fake = gen(fixed_noise)
        # take out (up to) 32 examples
        img_grid_real = torchvision.utils.make_grid(real[:100], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:100], normalize=True)

        writer_real.add_image("Real", img_grid_real, global_step=step)
        writer_fake.add_image("Fake", img_grid_fake, global_step=step)
        
        
        
        
        #BATCH LOSS-----
        
        #write gen_loss and critic_loss to tensorboard
        writer_loss.add_scalar("Generator loss Batch", loss_gen, global_step=step)
        writer_loss.add_scalar("Critic loss Batch", loss_critic, global_step=step)
        
        #BATCH LOSS-------
        
        #we will plot the gradient of critic output with respect to the input image
        #get the gradient of the critic output with respect to the input image
        gradient = torch.autograd.grad(
        inputs=real,
        outputs=critic_real,
        grad_outputs=torch.ones_like(critic_real),
        create_graph=True,
        retain_graph=True,
        )[0]
        #flatten the gradient
        gradient = gradient.view(gradient.shape[0], -1)
        #get the norm of the gradient
        gradient_norm = gradient.norm(2, dim=1)
        #write gradient norm to tensorboard
        writer_loss.add_scalar("Gradient norm Critic Real", gradient_norm.mean(), global_step=step)
        
        #----------------
        #we will plot the gradient of critic output with respect to the input image
        #get the gradient of the critic output with respect to the input image
        gradient = torch.autograd.grad(
        inputs=fake,
        outputs=critic_fake,
        grad_outputs=torch.ones_like(critic_fake),
        create_graph=True,
        retain_graph=True,
        )[0]
        #flatten the gradient
        gradient = gradient.view(gradient.shape[0], -1)
        #get the norm of the gradient
        gradient_norm = gradient.norm(2, dim=1)
        #write gradient norm to tensorboard
        writer_loss.add_scalar("Gradient norm Critic Fake", gradient_norm.mean(), global_step=step)
        
        #----------------
        #we will plot the gradient of genrator output with respect to the input 
        #we will plot the gradient of genrator output with respect to the input 
        #get the gradient of the generator output with respect to the input noise
        gradient = torch.autograd.grad(
        inputs=noise,
        outputs=gen_fake,
        grad_outputs=torch.ones_like(gen_fake),
        create_graph=True,
        retain_graph=True,
        )[0]
        #flatten the gradient
        gradient = gradient.view(gradient.shape[0], -1)
        #get the norm of the gradient
        gradient_norm = gradient.norm(2, dim=1)
        #write gradient norm to tensorboard
        writer_loss.add_scalar("Gradient norm Generator", gradient_norm.mean(), global_step=step)
        
        #----------------

        
        
        # we will plot the gradient penalty
        writer_loss.add_scalar("GP", gp, global_step=step)
        #we will analyze for vanishing gradient
        writer_loss.add_scalar("Critic Real", critic_real.mean(), global_step=step)
        writer_loss.add_scalar("Critic Fake", critic_fake.mean(), global_step=step)
        
        #get the gradient of the critic for the parameters weights of first layer
        #we will write the norm of the gardient of weights of the first layer of the critic
        for name, param in critic.named_parameters():
            if name == "disc.0.weight":
                writer_loss.add_scalar("Critic Gradient w.r.t 1st layer", param.grad.norm(), global_step=step)
            #also plot the norm of gradient of 2nd layer
            elif name == "disc.2.0.weight":
                writer_loss.add_scalar("Critic Gradient w.r.t 2nd layer", param.grad.norm(), global_step=step)
        
       
        
        
        

    step += 1
        
        #save the trained model
        #check if trained_model folder exists
    if not os.path.exists("trained_models"):
        os.mkdir("trained_models")
    
    #now trained_model folder exists
    if not os.path.exists("trained_models/"+model_name):
        os.mkdir("trained_models/"+model_name)
    #check if "trained_models/"+model_name     
    torch.save(gen.state_dict(), "trained_models/"+model_name+"/gen.pth")
    torch.save(critic.state_dict(), "trained_models/"+model_name+"/critic.pth")
    
    
    

