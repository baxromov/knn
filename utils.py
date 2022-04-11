import math


def get_euclidean(row1, row2):
    """
    Calculate the Euclidean distance between two rows
    :param row1:
    :type row1:
    :param row2:
    :type row2:
    :return:
    :rtype:
    """
    return math.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(row1, row2)]))


def get_cosine_sim(row1, row2):
    """
    Calculate the cosine similarity between two rows
    :param row1:
    :type row1:
    :param row2:
    :type row2:
    :return:
    :rtype:
    """
    return math.acos(
        sum([x1 * x2 for x1, x2 in zip(row1, row2)]) / (sum([i ** 2 for i in row1]) * sum([i ** 2 for i in row2]))
    )


def get_nearest(row, distance_measure, k):
    """
    k closest neighbors are selected for every point in the dev set.
    :param row:
    :type row:
    :param distance_measure:
    :type distance_measure:
    :param k:
    :type k:
    :return:
    :rtype:
    """
    return row[distance_measure][:k]


def get_dominant_class(df, neighbors):
    """
    Among the K nearest neighbors, the dominant class is elected and the data point is classified to belong to this class.
    :param df:
    :type df:
    :param neighbors:
    :type neighbors:
    :return:
    :rtype:
    """
    classes = df[df['seq'].isin(neighbors)]['class']
    return classes.value_counts().index[0]
