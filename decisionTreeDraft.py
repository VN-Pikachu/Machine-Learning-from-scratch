def returnLocalValue():
    from collections import Counter
    class Node():
        def __init__(self, value, feature):
            self.value = value
            self.feature = feature
    
    class Leaf():
        def __init__(self, value):
            self.value = value
    
    
    def split(data, labels, feature):
        values = set(tmp[feature] for tmp in data)
        subset_data, subset_labels = [], []
        for val in values:
            arr = []
            arr_labels = []
            for i in range(len(data)):
                tmp = data[i]
                if tmp[feature] == val:
                    arr.append(tmp)
                    arr_labels.append(labels[i])
            subset_data.append(arr)
            subset_labels.append(arr_labels)
        return subset_data, subset_labels
        
    def gini_impurity(labels):
        cnt = Counter(labels)
        ans, n = 1, len(labels)
        for name in cnt:
            ans -= (cnt[name] / n) ** 2
        return ans
    
    def weighted_information_gain(labels, subset_labels):
        ans = gini_impurity(labels)
        n = len(labels)
        for set_label in subset_labels:
            ans -= len(set_label) / n * gini_impurity(set_label)
        # print('Weighted', ans)
        return ans
    
    def best_feature(data, labels):
        best = max_gain = 0
        num_feature = len(data[0])
        for feature in range(num_feature):
            subset_data, subset_labels = split(data, labels, feature)
            gain = weighted_information_gain(labels, subset_labels)
            if gain > max_gain:
                best, max_gain = feature, gain
        print(data, labels, best, max_gain)
        return best, max_gain
    
    
    def tree(data, labels, value = None):
        feature, gain = best_feature(data, labels)
        if gain == 0:
            cnt = Counter(labels)
            return Leaf(max(cnt.items(), key = lambda x:x[1])[0])
        
        subset_data, subset_labels = split(data, labels, feature)
        node = Node(value, feature)
        node.children = [tree(subset_data[i], subset_labels[i], subset_data[i][0][feature]) for i in range(len(subset_data))]
        return node
    
    
    def search(data, tree):
        if isinstance(tree, Leaf):
            return tree.value
        # print(tree.feature, tree.value)
        category = data[tree.feature]
        #print(category)
        for subtree in tree.children:
            print(subtree.value)
            if subtree.value == category:
                return search(data, subtree)
        
        
    cars = [['F2', 'Lambo'], ['F3', 'Ferrari'],['F1', 'Lambo'], ['ST1', 'Ferrari']]
    car_labels = ['Exp', 'Med', 'Med', 'Exp']
    
    decisionTree = tree(cars, car_labels)
    print(decisionTree.value, decisionTree.feature)
    print(search(['F3', 'Ferrari'], decisionTree))

