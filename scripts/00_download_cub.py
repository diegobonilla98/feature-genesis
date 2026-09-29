from feature_genesis.data.cub_download import ensure_cub_dataset


def main():
    cub_root = ensure_cub_dataset()
    print(cub_root)


if __name__ == "__main__":
    main()
