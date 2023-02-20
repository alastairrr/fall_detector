from fall_lstm import LSTMFallDetector


def main():
    model = LSTMFallDetector()
    model.getModelSummary()
    model.modelFit()

if __name__ == "__main__":
    main()