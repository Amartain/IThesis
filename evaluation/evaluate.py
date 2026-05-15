from data.paths import get_dataset_dirs

def setup_evaluation(model_path, dataset_choice, max_dim_choice, batch_size, no_workers):
    print("Evaluation Started With Parameters: ")
    print(dataset_choice, max_dim_choice, batch_size, no_workers)

    print(get_dataset_dirs(dataset_choice))
