import numpy as np
import matplotlib.pyplot as plt
from Training import train_model, load_mnist_data
from Networks import MNISTClassifierTF2  # Import other models as needed


# Model choice
selected_model_name = input(
    "Choose model (LR_For_Loop / LR_Sklearn / LR_TF_Basic / LR_TF_Advance / SLR_in_Tensor_env / MNISTClassifierTF2): ")

if selected_model_name == "MNISTClassifierTF2":
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_mnist_data()
    model = train_model(selected_model_name, train_images, train_labels, val_images, val_labels)
    model.evaluate(test_images, test_labels)
    model.plot_loss()

else:
    # Generate synthetic regression data
    observations = 1000
    xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
    zs = np.random.uniform(-10, 10, (observations, 1))
    inputs = np.column_stack((xs, zs))

    noise = np.random.uniform(-1, 1, size=(observations, 1))
    targets = 2 * xs - 3 * zs + 5 + noise

    # Save data
    #np.savez('TF_intro', inputs=inputs, targets=targets)

    # Train the selected model
    model = train_model(selected_model_name, inputs, targets)

    # Make predictions and display results
    test_input = np.array([[5, -3]], dtype=np.float32)
    prediction = model.predict(test_input)
    print(f"\nPrediction for input [5, -3] is: {prediction} and Exact value is: {2*test_input[0][0] - 3*test_input[0][1] + 5}")

    # Visualize the data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, zs, targets)
    ax.set_xlabel('xs')
    ax.set_ylabel('zs')
    ax.set_zlabel('Targets')
    ax.view_init(azim=100)
    plt.show()