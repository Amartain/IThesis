from configurations.configure import start

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    start()
