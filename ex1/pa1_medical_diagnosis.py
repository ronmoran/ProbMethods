"""
67800 - Probabilistic Methods in AI
Spring 2022
Programming Assignment 1 - Bayesian Networks
(Complete the missing parts (TODO))
"""

import numpy as np 
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.special import logsumexp
import seaborn as sns


def load_model(model_file):
    """
    Loads a default Bayesian network with latent variables
    """

    with open(model_file + '.pkl', 'rb') as infile:
        model = pkl.load(infile, encoding='bytes')

    return model


def load_data(data_file):
    """
    Loads data files
    """

    with open(data_file + '.pkl', 'rb') as infile:
        mat = pkl.load(infile, encoding='bytes')
    return mat


def accuracy(pred, true):
    """
    Calculates the full and top1 accuracy (depending on the input's shape).
    """
    return np.abs(pred == true).mean()


class NaiveBayes:
    def __init__(self, cpds, diseases, symptoms):
        self.cpds = cpds
        self.diseases = diseases
        self.symptoms = symptoms

    def get_p_D(self, disease):
        """
        Get the prior probability for variable D to get the value "disease".
        """
        return self.cpds['D'][disease].values

    def get_p_S_cond_D(self, disease):
        """
        Get the conditional probabilities of the symptoms to take the value 1, given D = disease.
        """
        return self.cpds['S_given_D'][disease].values

    def get_expectation_S_cond_D(self, disease):
        """
        TODO. Calculate the conditional expectation E(Sj | disease) for j=1,...,m
        :param disease: disease name (string)
        :return: Array of m conditional expectation E(Sj | disease) (for each symptom)
        """
        return self.get_p_S_cond_D(disease)

    def get_log_p_S_joint_D(self, data, disease):
        """
        TODO. Compute the joint log probability log P(S, disease)
        :param data: Row vectors of data points S (n x m)
        :param disease: disease name (string)
        :return: Array of n log probability values log P(S, disease) (for each data point)
        """
        probs = np.where(data, self.get_p_S_cond_D(disease), 1 - self.get_p_S_cond_D(disease))
        return np.log(probs).sum(axis=1) + np.log(self.get_p_D(disease))

    def get_log_p_S(self, data) -> np.ndarray:
        """
        TODO. Compute the marginal log probabilities (log-likelihood): log P(S = data)
        :param data: Row vectors of data points S (n x m)
        :return: Array of n log probability values log P(S = data) (for each data point)
        """
        data_joint_prob = np.empty((data.shape[0], len(self.diseases)), dtype=float)
        for col_index, disease in enumerate(self.diseases):
            data_joint_prob[:, col_index] = self.get_log_p_S_joint_D(data, disease)
        return logsumexp(data_joint_prob, axis=1)


    def get_p_D_given_S(self, data):
        """
        TODO. Calculate the conditional probability p(disease | S = data[i]) for each data point and disease
        :param data: Row vectors of data points S (n x m)
        :return: array of K probabilities p(disease | S = data) (for each disease)
        """
        pass

    def predict(self, data):
        """
        TODO. Return the prediction for each data point according to the rule
                        d_hat = arg max {p(disease | s) over disease in diseases}
        :param data: Row vectors of data points S (n x m)
        :return: array of n predicted labels (for each data point)
        """
        pass


def q_2():
    """
    Plots the clustermap of the conditional expectation of the symptoms given each disease.
    """
    plt.figure()
    e_sym_given_d = np.zeros((M, K))
    for k in range(K):
        e_sym_given_d[:, k] = nb.get_expectation_S_cond_D(diseases[k])
    sns.clustermap(e_sym_given_d, xticklabels=diseases, yticklabels=symptoms, cmap='RdYlBu_r')
    plt.tight_layout()
    plt.savefig('q2', bbox_inches='tight')
    plt.show()
    plt.close()


def q_3():
    """
    Loads the data and plots the histograms. Rest is TODO.
    Your job is to compute the validation_marginal_log_likelihood, real_marginal_log_likelihood and corrupt_marginal_log_likelihood below.
    """
    # get data
    mat = load_data('data/test_validation')
    validation_data = mat['validation'].values
    test_data = mat['test'].values
    validation_marginal_log_likelihood = nb.get_log_p_S(validation_data)
    avg = validation_marginal_log_likelihood.mean()
    std = validation_marginal_log_likelihood.std()
    marginal_log_test = nb.get_log_p_S(test_data)
    corrupt_indices = marginal_log_test < avg - 3 * std
    real_marginal_log_likelihood = marginal_log_test[~corrupt_indices]
    corrupt_marginal_log_likelihood = marginal_log_test[corrupt_indices]

    '''
    TODO. Calculate marginal_log_likelihood on validation data, define prediction rule, 
            and calculate marginal_log_likelihood of test samples classified as real and as corrupted.
    '''

    # plot histograms
    plt.figure()
    plt.title('Histogram of marginal log-likelihood')
    mi = np.min([corrupt_marginal_log_likelihood.min(), real_marginal_log_likelihood.min(), validation_marginal_log_likelihood.min()])
    _, bins, _ = plt.hist(validation_marginal_log_likelihood, label='validation data', bins=np.arange(mi-10, 0, 1))
    plt.hist(real_marginal_log_likelihood, label='real test data', bins=bins)
    plt.hist(corrupt_marginal_log_likelihood, label='corrupted test data', bins=bins)
    plt.xlabel('marginal log-likelihood')
    plt.ylabel('frequency')
    plt.legend()
    plt.savefig('q3_hist', bbox_inches='tight')
    plt.show()
    plt.close()



def q_4():
    """
    Loads the data and calculate the top1-accuracy according to the conditional expectations of the Naive Bayes model.
    Your job is to implement the predict function for the Naive Bayes model (no need to change this function).
    """

    mat = load_data('data/single_label_data')
    data, labels = mat['data'].values, mat['labels'].values
    pred = nb.predict(data)
    print(f'top1-accuracy for Naive Bayes = {accuracy(pred, labels[:, 0])}')


def q_5():
    """
    Loads the data and calculate the top1-accuracy according to the conditional expectations of the Naive Bayes model.
    Your job is to implement the predict function for the Naive Bayes model (no need to change this function).
    """

    mat = load_data('data/mul_label_data')
    data, labels = mat['data'].values, mat['labels'].values
    pred = nb.predict(data)
    mul_pred = np.zeros(labels.shape)
    mul_pred[np.arange(labels.shape[0]), pred] = 1
    print(f'full-accuracy for Naive Bayes = {accuracy(mul_pred, labels)}')


def main():
    global nb, diseases, symptoms, M, K
    nb_cpds = load_model('data/nb')
    symptoms = nb_cpds['symptoms']
    diseases = nb_cpds['diseases']
    nb = NaiveBayes(nb_cpds, diseases, symptoms)
    K, M = len(diseases), len(symptoms)

    q_2()
    q_3()
    # q_4()
    # q_5()


if __name__== '__main__':
    main()
