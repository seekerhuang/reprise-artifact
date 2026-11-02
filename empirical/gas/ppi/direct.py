def train(model, loader, criterion, optimizer, max_steps):
    model.train()
    for _ in range(2):
        for index, (batch, core, *coordinates) in enumerate(loader):
            x = batch.x.to(model.device)
            matrix = batch.adj_t.to(model.device)
            labels = batch.y[:core].to(model.device)
            host_mask = batch.train_mask[:core]
            if not int(host_mask.sum()):
                continue
            mask = host_mask.nonzero(as_tuple=True)[0].to(model.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, matrix, core, *coordinates)
            loss = criterion(logits[mask], labels[mask])
            loss.backward()
            optimizer.step()
            if index + 1 >= max_steps and index + 1 < len(loader):
                break
    optimizer.zero_grad(set_to_none=True)

