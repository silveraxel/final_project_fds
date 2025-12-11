from init import *


if MODALITY == 'training':
    best_val_rmse = float('inf')
    best_train_rmse = float('inf')
    patience_counter = 0
    metrics = []
    log_parameters(current_params)

    if LOAD_MODEL:
        model_weights = torch.load(MODEL_PATH)
        model.load_state_dict(model_weights)
        print('Loading the model '+str(MODEL_PATH))
    else:
        print('Creating/Overwriting the model '+str(MODEL_PATH))
    
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train()
        train_rmse = test(train_loader)
        val_rmse = test(val_loader)
        
        train_val_gap = val_rmse - train_rmse
        metrics.append({
            'epoch': epoch,
            'loss': loss,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse
        })

        if USE_LR_SCHEDULER:
            scheduler.step(val_rmse)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_train_rmse = train_rmse
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            improvement_marker = "✓"
        else:
            patience_counter += 1
            improvement_marker = ""
        print(f'Patience Counter: {patience_counter:03d} \n')
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
            f'Train: {train_rmse:.4f}, Val: {val_rmse:.4f}, '
            f'Gap: {train_val_gap:.4f}')
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f'\n✓ Early stopping at epoch {epoch}')
            break
    plot_metrics(metrics, current_params)
    
elif MODALITY == 'inference':
# Load best model
    model_weights = torch.load(MODEL_PATH, weights_only = True)
    model.load_state_dict(model_weights)
    movies_df = pd.read_csv(movie_path)



    # Run sample predictions with visualization
    #errors, actual, predicted = evaluate_sample_predictions(current_params, movies_df, n_users=80, n_samples_per_user=30)
    all_errors, all_actual, all_predicted, all_actual_binary, all_predicted_binary = evaluate_sample_predictions(current_params, movies_df, n_users=80, n_samples_per_user=40, rating_threshold=4.0)

else:
    print('Internal error no valid modality has been defined. Exiting the script')
    exit(0)


