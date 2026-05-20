from training.train_sms_model import train_sms_model
from training.train_url_model import train_url_model


def main():
    train_sms_model()
    train_url_model()
    print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()