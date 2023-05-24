import config
import numpy as np
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


def initialize():
    population = []
    for i in range(0, config.POPULATION_SIZE):
        solution = []
        for j in range(0, config.NUM_FEATURES):
            solution.append(random.uniform(0, 1)*math.pi*2)
        population.append(solution)
    return population


def observe(population):
    population_observed = []
    for solution in population:
        solution_observed = []
        for feature in solution:
            if math.pow(math.cos(feature), 2) > random.uniform(0, 1):
                solution_observed.append(0)
            else:
                solution_observed.append(1)
        population_observed.append(solution_observed)
    return population_observed


def evaluate_fitness(population_observed, X_train, X_val, y_train, y_val):
    scores = []
    fitness_scores = []
    features_num = []

    for solution_observed in population_observed:
        drop_index = [feature_index for feature_index, feature in enumerate(solution_observed) if feature == 0]
        X_train_drop, X_val_drop = X_train.drop(X_train.columns[drop_index], axis=1), X_val.drop(X_val.columns[drop_index], axis=1)

        classifier = SVC(kernel='linear').fit(X_train_drop, y_train)
        train_score = f1_score(y_val, classifier.predict(X_val_drop), average='weighted')
        
        scores.append(train_score)
        fitness_scores.append(train_score + (1-X_train_drop.shape[1]/config.NUM_FEATURES))
        features_num.append(X_train_drop.shape[1])
        
    return scores, fitness_scores, features_num


def evaluate_performance(population_observed, X_train, X_test, y_train, y_test):
    f1_scores = []
    acc_scores = []

    for solution_observed in population_observed:
        drop_index = [feature_index for feature_index, feature in enumerate(solution_observed) if feature == 0]
        X_train_drop, X_test_drop = X_train.drop(X_train.columns[drop_index], axis=1), X_test.drop(X_test.columns[drop_index], axis=1)

        classifier = SVC(kernel='linear').fit(X_train_drop, y_train)
        pred = classifier.predict(X_test_drop)
        f1_scores.append(f1_score(y_test, pred, average='weighted'))
        acc_scores.append(accuracy_score(y_test, pred))

    return f1_scores, acc_scores


def mutate(population):
    population_mutated = []

    for solution_index, solution in enumerate(population):
        index = list(range(0, config.POPULATION_SIZE))
        index.remove(solution_index)
        rand_indexes = random.sample(index, 3)
        Rand1 = population[rand_indexes[0]]
        Rand2 = population[rand_indexes[1]]
        Rand3 = population[rand_indexes[2]]

        solution_mutated = []
        for feature_index in range(0, config.NUM_FEATURES):
            solution_mutated.append(Rand1[feature_index] + config.F * (Rand2[feature_index] - Rand3[feature_index]))
        population_mutated.append(solution_mutated)

    return population_mutated


def crossover(population, population_mutated):
    population_crossed = []

    for solution, solution_mutated in zip(population, population_mutated):
        Irand = int(1 + random.uniform(0, 1) * config.NUM_FEATURES)

        solution_crossed = []
        for feature_index in range(0, config.NUM_FEATURES):
            if random.uniform(0, 1) <= config.CR and feature_index == Irand:
                solution_crossed.append(solution_mutated[feature_index])
            else:
                solution_crossed.append(solution[feature_index])
        population_crossed.append(solution_crossed)

    return population_crossed


def select(population, population_observed, scores, fitness_scores, features_num, population_crossed, population_crossed_observed, scores_crossed, fitness_scores_crossed, features_num_crossed):
    population_selected = []
    population_observed_selected = []
    scores_selected = []
    fitness_scores_selected = []
    features_num_selected = []

    for solution, solution_observed, score, fitness_score, feat_num, solution_crossed, solution_observed_crossed, score_crossed, fitness_score_crossed, feat_num_crossed in zip(population, population_observed, scores, fitness_scores, features_num, population_crossed, population_crossed_observed, scores_crossed, fitness_scores_crossed, features_num_crossed):

        if fitness_score < config.ELITISM and fitness_score_crossed > fitness_score:
            population_selected.append(solution_crossed)
            population_observed_selected.append(solution_observed_crossed)
            scores_selected.append(score_crossed)
            fitness_scores_selected.append(fitness_score_crossed)
            features_num_selected.append(feat_num_crossed)
        else:
            population_selected.append(solution)
            population_observed_selected.append(solution_observed)
            scores_selected.append(score)
            fitness_scores_selected.append(fitness_score)
            features_num_selected.append(feat_num)

    return population_selected, population_observed_selected, scores_selected, fitness_scores_selected, features_num_selected


def write_output(test_f1_scores, test_acc_scores, fold_features_num, fold_population_observed, fold_time, run_time):
    f = open(config.OUTPUT_FILEPATH + config.DATASET_FILEPATH.split("/")[-1][:-4] + "_QDE-SVM_" +  time.strftime("%Y%m%d-%H%M%S") + ".txt", 'w')

    f.write('DATASET: ' + config.DATASET_FILEPATH.split("/")[-1][:-4] + '\n')
    f.write('NUM_FEATURES: ' + str(config.NUM_FEATURES) + '\n')
    f.write('POPULATION_SIZE: ' + str(config.POPULATION_SIZE) + '\n')
    f.write('ITERATION_NUM: ' + str(config.ITERATION_NUM) + '\n')
    f.write('FOLDS: ' + str(config.K) + '\n')
    f.write('ELITISM: ' + str(config.ELITISM) + '\n')
    f.write('F: ' + str(config.F) + '\n')
    f.write('CR: ' + str(config.CR) + '\n\n')
    
    f.write('Test Accuracy: ' + str(np.mean([np.mean(acc) for acc in test_acc_scores])) + '\n')
    f.write('Test F1 Score: ' + str(np.mean([np.mean(score) for score in test_f1_scores])) + '\n')
    f.write('Average Features No.: ' + str(np.mean([np.mean(num) for num in fold_features_num])) + '\n')
    f.write('Total time taken: ' + str(run_time) + ' seconds\n\n')

    f.write('Fold\tTest_Acc\tTest_F1\tMean_Features_Num\tFold_time(secs)\n')
    for k in range(0, config.K):
        f.write(str(k+1)+'\t')
        f.write(str(np.mean(test_acc_scores[k]))+'\t')
        f.write(str(np.mean(test_f1_scores[k]))+'\t')
        f.write(str(np.mean(fold_features_num[k]))+'\t')
        f.write(str(fold_time[k])+'\n')

    f.write('\nFold\tSolution\tTest_Acc\tTest_F1\tFeatures_Num\tObserved_Solution\n')
    for k in range(0, config.K):
        for solution in range(0, config.POPULATION_SIZE):
            f.write(str(k+1)+'\t')
            f.write(str(solution+1)+'\t')
            f.write(str(test_acc_scores[k][solution])+'\t')
            f.write(str(test_f1_scores[k][solution])+'\t')
            f.write(str(fold_features_num[k][solution])+'\t')
            f.write(''.join(str(feature) for feature in fold_population_observed[k][solution]) + '\n')


def QDESVM():
    start_run = time.time()

    test_f1_scores = []
    test_acc_scores = []
    fold_time = []
    fold_population_observed = []
    fold_features_num = []

    for k in range(config.K): 
        print("Fold: ", k+1)
        start_fold = time.time()

        # split train data into train and val
        X_train, X_val, y_train, y_val = train_test_split(config.X_trains[k], config.y_trains[k], test_size=0.2, random_state=42, stratify=config.y_trains[k])

        # initialization
        population = initialize()
        population_observed = observe(population)
        if k == 0:
            start = time.time()
            scores, fitness_scores, features_num = evaluate_fitness(population_observed, X_train, X_val, y_train, y_val)
            print("Estimated time: " + str(((time.time() - start) * config.ITERATION_NUM * config.K) / 60) + " minutes")
        else:
            scores, fitness_scores, features_num = evaluate_fitness(population_observed, X_train, X_val, y_train, y_val)

        # update elitism threshold according to initial population score and ideal portion of features
        ideal_score = np.mean(scores)+0.1
        if ideal_score >= 1.0:
            ideal_score = 1.0
        config.ELITISM = ideal_score + (1 - config.IDEAL_PORTION_FEATURES)

        print("Initialization - Mean train score: " + str(np.mean(scores)) + " Mean number of features: " + str(np.mean(features_num)))
        
        # start QDE
        for iter in range(0, config.ITERATION_NUM):
            # mutation
            population_mutated = mutate(population)
            # crossover
            population_crossed = crossover(population, population_mutated)
            population_crossed_observed = observe(population_crossed)
            scores_crossed, fitness_scores_crossed, features_num_crossed = evaluate_fitness(population_crossed_observed, X_train, X_val, y_train, y_val) 
            # selection
            population, population_observed, scores, fitness_scores, features_num = select(population, population_observed, scores, fitness_scores, features_num, population_crossed, population_crossed_observed, scores_crossed, fitness_scores_crossed, features_num_crossed)

            print("Iteration " + str(iter+1) + " - Mean train score: " + str(np.mean(scores)) + " Mean number of features: " + str(np.mean(features_num)))

        fold_time.append(time.time() - start_fold)
        fold_population_observed.append(population_observed)
        fold_features_num.append(features_num)

        # performance evaluation
        f1_scores, acc_scores = evaluate_performance(population_observed, config.X_trains[k], config.X_tests[k], config.y_trains[k], config.y_tests[k])
        test_f1_scores.append(f1_scores)
        test_acc_scores.append(acc_scores)

    # write output
    write_output(test_f1_scores, test_acc_scores, fold_features_num, fold_population_observed, fold_time, time.time()-start_run)


if __name__ == "__main__":
    QDESVM()