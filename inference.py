def sample(self, text_embeddings, num_steps=1000):
    self.pose_transformer.eval()
    batch_size = text_embeddings.shape[0]
    pose_dim = self.pose_transformer.embedding_dim

    # Start with pure Gaussian noise
    x_t = torch.randn((batch_size, 1, pose_dim), device=self.device)

    for t in reversed(range(num_steps)):
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        predicted_noise = self.text_cross_transformer(x_t, text_embeddings)

        # Remove predicted noise
        alpha_bar_t = self.noise_scheduler.alpha_bars[t].view(-1, 1, 1).to(x_t.device)
        x_t = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

    return x_t  # Generated pose sequence