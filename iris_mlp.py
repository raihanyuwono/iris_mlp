import math
import random
import copy
import matplotlib.pyplot as plt

def __convertDataStructure__(attr) :
    try :
        attr = float(attr)
    except :
        if attr == 'Iris-setosa' :
            attr = 0
        elif attr == 'Iris-versicolor' :
            attr = 0.5
        else : # Iris-virginica
            attr = 1
    finally :
        return attr 

def __readData__() :
    data = []
    try :
        file = open('iris.data')
        for line in file :
            data.append([__convertDataStructure__(attribute) for attribute in line.strip().split(',')])
    finally :
        file.close()
    return data

def __randomWeight__() :
    global layers, neurons
    hiddenLayers = 1
    layers = hiddenLayers + 1
    outputNeuron = 1
    neurons = [random.randint(1, 10) for _ in range(hiddenLayers)]
    neurons.append(outputNeuron)
    weight = [[] for _ in range(layers)]
    for layer in range(layers) :
        for _ in range(neurons[layer]) :
            if layer == 0 :
                weight[layer].append([random.random() for _ in range(len(dataSet[0]))])
            else :
                weight[layer].append([random.random() for _ in range(neurons[layer - 1] + 1)])
    return weight
    # access neurons -> neurons[layer]
    # access weight -> weight[layer][neuron][i]

def __initData__() :
    global dataSet, learningRate, epoch
    learningRate = 0.1
    epoch = 100
    dataSet = __readData__()
    return __randomWeight__()

def __plotGraphics__(*errors) :
    for error in errors :
        # print(error[1])
        # for i in range(len(error[0]['Training'])) :
        #     print('Training : %.2f\tValidation : %.2f' % (error[0]['Training'][i], error[0]['Validation'][i]))
        plt.plot(error[0]['Training'], label=error[1] + ' Training')
        plt.plot(error[0]['Validation'], label=error[1] + ' Validation')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

def __errorFunction__(fact, prediction) :
    avg = 0.0
    for pred in prediction :
        avg += ((fact - pred) ** 2) / 2
    return avg / len(prediction)

def __activationFunction__(target, mode='sigmoid') :
    if mode == 'sigmoid' :
        return 1 / (1 + math.exp(-target))
    else :
        return math.exp(-target) / (1 + math.exp(-target))

def __targetFunction__(data, weight) :
    sum = 0.0
    for i in range(len(data) - 1) :
        sum += data[i] * weight[i]
    sum += weight[len(data) - 1]
    return sum

def __deltaFunctionFWD__(attr, prediction, fact) :
    return (prediction - fact) * (1 - prediction) * prediction * attr

def __updateFeedForward__(data, weight, prediction, fact) :
    for i in range(len(data) - 1) :
        weight[i] -= learningRate *__deltaFunctionFWD__(data[i], prediction, fact)
    return weight

def __deltaFunctionBP__(tau, data) :
    return tau * data

def __updateBackPropagation__(data, weight, prediction):
    tau = [[] for _ in range(layers)]
    for layer in range(layers - 1, -1, -1) :
        # print(prediction[layer])
        for neuron in range(neurons[layer]) :
            pred = prediction[layer][neuron]
            if layer == layers - 1 : # Output Layer
                fact = data[len(data) - 1]
                tau[layer].append((fact - pred) * (1 - pred) * pred)
            else : # Hidden Layers
                total = 0.0
                for nextNeuron in range(neurons[layer + 1]) :
                    total += tau[layer + 1][nextNeuron] * weight[layer + 1][nextNeuron][neuron]
                tau[layer].append(total * pred * (1 - pred))
            dataUsed = data[0:len(data) - 1] if layer == 0 else prediction[layer - 1]
            for dataPrev in range(len(dataUsed)) : # Update Weight
                weight[layer][neuron][dataPrev] -= learningRate * __deltaFunctionBP__(tau[layer][neuron], dataUsed[dataPrev])
            weight[layer][neuron][len(dataUsed)] -= learningRate * __deltaFunctionBP__(tau[layer][neuron], 1)
    return weight

def __training__(data, weight, error, mode='FWD') :
    prediction = [[] for _ in range(layers)]
    fact = data[len(data) - 1]
    for layer in range(layers) :
        for neuron in range(neurons[layer]) :
            dataUsed = data if layer == 0 else prediction[layer - 1]
            target = __targetFunction__(dataUsed, weight[layer][neuron])
            prediction[layer].append(__activationFunction__(target))
            # print(prediction[layer][neuron])
            if mode == 'FWD' :
                weight[layer][neuron] = __updateFeedForward__(dataUsed, weight[layer][neuron], prediction[layer][neuron], fact)
    # print(prediction)
    if mode == 'BP' :
        weight = __updateBackPropagation__(data, weight, prediction)
    # print(mode)
    # for temp in weight :
    #     print(temp)
    return weight, error + __errorFunction__(fact, prediction[layers - 1])

def __validation__(data, weight) :
    prediction = [[] for _ in range(layers)]
    fact = data[len(data) - 1]
    for layer in range(layers) :
        for neuron in range(neurons[layer]) :
            dataUsed = data if layer == 0 else prediction[layer - 1]
            target = __targetFunction__(dataUsed, weight[layer][neuron])
            prediction[layer].append(__activationFunction__(target))

    return __errorFunction__(fact, prediction[layers - 1])

def __crossValidation__(n_data, weight) :
    size = len(dataSet) // n_data
    weightFeedForward = copy.deepcopy(weight)
    weightBackPropagation = copy.deepcopy(weight)
    nTraining = n_data * (n_data -1) * size
    nValidation = n_data * size
    subData = [dataSet[start:start + size] for start in range(0, len(dataSet), size)]
    errorFWD =  {'Training':[], 'Validation':[]}
    errorBP = {'Training':[], 'Validation':[]}
    for _ in range(epoch) :
        errorTrainingFWD, errorValidationFWD = (0.0, 0.0)
        errorTrainingBP, errorValidationBP = (0.0, 0.0)
        for dataValidation in subData :
            for dataTraining in subData :
                if dataTraining != dataValidation :
                    for data in dataTraining :
                        weightFeedForward, errorTrainingFWD = __training__(data, weightFeedForward, errorTrainingFWD, mode='FWD')
                        weightBackPropagation, errorTrainingBP = __training__(data, weightBackPropagation, errorTrainingBP, mode='BP')
            for data in dataValidation :
                errorValidationFWD += __validation__(data, weightFeedForward)
                errorValidationBP += __validation__(data, weightBackPropagation)
        # print(weight)
        # print('FeedForward\n', weightFeedForward, '\n')
        # print('BackPropagation\n', weightBackPropagation, '\n\n')
        errorFWD['Training'].append(errorTrainingFWD / nTraining)
        errorFWD['Validation'].append(errorValidationFWD / nValidation)
        errorBP['Training'].append(errorTrainingBP / nTraining)
        errorBP['Validation'].append(errorValidationBP / nValidation)
    
    __plotGraphics__([errorFWD, 'FeedForward'], [errorBP, 'BackPropagation'])

def __main__() :
    weight = __initData__()
    __crossValidation__(5, weight)
    # print('Layer = ', layers)
    # print('Neuron = ', neurons)
    # for layer in range(layers) :
    #     print(weightFeedForward[layer])

__main__()