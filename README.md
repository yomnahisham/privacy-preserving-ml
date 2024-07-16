# Privacy-Preserving Machine Learning Implementation
A script for privacy-preserving machine learning implementation, allowing secure data analysis without exposing individual data points by implementing differential privacy techniques to ensure data security and privacy.

#### Table of Contents:
- [Introduction](#introduction)
- [Features](#features)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

#### Introduction:
This project aims to provide a robust tool for performing data mining tasks while preserving the privacy of individual data points. By adding noise to the data, we can ensure that sensitive information is protected while still enabling meaningful analysis.

#### Features:
- **Differential Privacy:** Implements the Laplace mechanism for differential privacy to protect individual data points.
- **Data Mining:** Supports various data mining tasks such as classification and clustering.
- **Versatility:** Tested on multiple datasets to demonstrate its applicability.

#### Datasets:
The tool has been tested on the following datasets:
- **Iris dataset:** A classic dataset for classification tasks.
- **Wine dataset:** Another classification dataset to test the tool's robustness.
- **Synthetic Adult Census Income dataset:** A well-known dataset from UCI, used to demonstrate the tool’s application to more sensitive data.

#### Installation:
To get started with the Privacy-Preserving Machine Learning Implementation, follow these steps:

1. Clone the repository:
   ```sh
   git clone [[GitHub Repository Link](https://github.com/yomnahisham/privacy-preserving-ml)]
   cd privacy-preserving-ml
   ```

2. Install the required libraries:
   ```sh
   pip install pandas scikit-learn matplotlib jupyterlab numpy
   ```

#### Usage:
1. Run the script: 
   ```sh
   python3 laplace.py
   ```
2. Modify the dataset_name variable in the laplace.py file to choose between ‘iris’, ‘wine’, ‘adult’, or your custom dataset.
3. Adjust the epsilon parameter in the laplace.py file to vary the privacy level.

#### Results:
The tool has been evaluated on the Iris, Wine, and Adult Census Income datasets. The results demonstrate that our privacy-preserving techniques effectively protect sensitive information while maintaining a reasonable level of data utility.

#### Contributing:
Contributions are welcome! Please fork this repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

Feel free to reach out with any questions or feedback.
