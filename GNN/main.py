from init import *


if MODALITY == 'training':
    best_val_rmse = float('inf')
    best_train_rmse = float('inf')
    val_rmse_history = []
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
        val_rmse_history.append(val_rmse)
        train_val_gap = val_rmse - train_rmse

        if len(val_rmse_history) >= EARLY_STOPPING_WINDOW:
            current_window_mean = sum(val_rmse_history[-EARLY_STOPPING_WINDOW:]) / EARLY_STOPPING_WINDOW
        else:
            # If we don't have enough history yet, use all available
            current_window_mean = sum(val_rmse_history) / len(val_rmse_history)

        if USE_LR_SCHEDULER:
            #scheduler.step(val_rmse)
            scheduler.step(current_window_mean)

        if current_window_mean < best_val_rmse:
            best_val_rmse = current_window_mean
            best_train_rmse = train_rmse
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            improvement_marker = "✓"
            print(f'New best window mean: {current_window_mean:.4f} (window size: {min(len(val_rmse_history), EARLY_STOPPING_WINDOW)})')
        else:
            patience_counter += 1
            improvement_marker = ""
        
        metrics.append({
            'epoch': epoch,
            'loss': loss,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'current_window_mean' : current_window_mean
        })
        print(f'Patience Counter: {patience_counter:03d} / {EARLY_STOPPING_PATIENCE}')
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
        f'Train: {train_rmse:.4f}, Val: {val_rmse:.4f}, '
        f'Gap: {train_val_gap:.4f}, Window Mean: {current_window_mean:.4f} {improvement_marker}\n')

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f'\n✓ Early stopping at epoch {epoch}')
            break
        
    plot_metrics(metrics, current_params)
    
elif MODALITY == 'inference':
# Load best model
    model_weights = torch.load(MODEL_PATH, weights_only = True)
    model.load_state_dict(model_weights)
    movies_df = pd.read_csv(movie_path)


    #GNN VISUALIZER
    #visualizer = EdgeInfluenceVisualizer(model, test_data, movies_df, device=device)
    #model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    #visualizer.create_influence_visualization_suite(f'./influence_viz_{model_name}')

    all_errors, all_actual, all_predicted, all_actual_binary, all_predicted_binary = evaluate_sample_predictions(current_params, movies_df, n_users=80, n_samples_per_user=30, rating_threshold=3.5)

else:
    print('Internal error no valid modality has been defined. Exiting the script')
    exit(0)


