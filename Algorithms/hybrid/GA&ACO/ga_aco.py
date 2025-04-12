# Importação das bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# Parâmetros do algoritmo híbrido GA-ACO
POPULATION_SIZE = 30           # Tamanho da população
CHROMOSOME_LENGTH = None       # Será definido pelo número de features
MAX_GENERATIONS = 50           # Número máximo de gerações
MUTATION_RATE = 0.05           # Taxa de mutação para o GA
CROSSOVER_RATE = 0.8           # Taxa de cruzamento para o GA
TOURNAMENT_SIZE = 3            # Tamanho do torneio para seleção
ACO_RATIO = 0.3                # Proporção da população que usa ACO (30%)
MIN_FEATURES = 2               # Número mínimo de features a serem selecionadas

# Parâmetros específicos do ACO
ALPHA = 1.0                    # Importância do feromônio
BETA = 2.0                     # Importância da informação heurística
EVAPORATION_RATE = 0.1         # Taxa de evaporação do feromônio
Q = 1.0                        # Constante de depósito de feromônio


class GAACO:
    """Algoritmo híbrido Genético + ACO para seleção de features."""
    
    def __init__(self, X, y, classifier=None, cv=5):
        """
        Inicializa o algoritmo híbrido GA-ACO.
        
        Params:
            X: Conjunto de dados (amostras x features)
            y: Classes/rótulos
            classifier: Classificador para avaliar os subconjuntos de features
            cv: Número de folds para validação cruzada
        """
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.cv = cv
        
        # Define o tamanho do cromossomo baseado no número de features
        global CHROMOSOME_LENGTH
        CHROMOSOME_LENGTH = self.n_features
        
        # Define o classificador padrão se nenhum for fornecido
        if classifier is None:
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.classifier = classifier
            
        # Melhor solução global
        self.g_best = None
        self.g_best_fitness = -np.inf
        
        # Histórico da otimização
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.feature_count_history = []
        
        # Matriz de feromônio para o ACO (inicializada com valores iguais)
        self.pheromone = np.ones(self.n_features)
        
        # Importância heurística de cada feature (pode ser baseada em filtros)
        self.heuristic = np.ones(self.n_features)
        self._init_heuristic()
        
    def _init_heuristic(self):
        """
        Inicializa a informação heurística para cada feature.
        Neste caso simples, usamos correlação com a classe alvo.
        """
        for i in range(self.n_features):
            # Correlação absoluta entre a feature e a classe
            corr = np.abs(np.corrcoef(self.X[:, i], self.y)[0, 1])
            if np.isnan(corr):  # Em caso de erro/NaN
                corr = 0.1
            self.heuristic[i] = max(0.1, corr)  # Evita valores muito pequenos
    
    def initialize_population(self):
        """Inicializa a população com cromossomos binários aleatórios."""
        # Garantir que cada solução tenha pelo menos MIN_FEATURES ativas
        population = np.zeros((POPULATION_SIZE, self.n_features), dtype=np.int8)
        
        for i in range(POPULATION_SIZE):
            # Seleciona aleatoriamente algumas features (pelo menos MIN_FEATURES)
            n_selected = random.randint(MIN_FEATURES, self.n_features)
            selected_features = random.sample(range(self.n_features), n_selected)
            population[i, selected_features] = 1
            
        return population
    
    def fitness_function(self, chromosome):
        """
        Calcula o fitness usando validação cruzada no classificador com as features selecionadas.
        Inclui penalidade para muitas features (promove parcimônia).
        """
        # Conta quantas features foram selecionadas
        n_selected_features = np.sum(chromosome)
        
        # Se menos que o mínimo de features, retorna fitness muito baixo
        if n_selected_features < MIN_FEATURES:
            return -np.inf
        
        # Seleciona apenas as colunas (features) marcadas como 1 no cromossomo
        selected_features_idx = np.where(chromosome == 1)[0]
        X_selected = self.X[:, selected_features_idx]
        
        # Avalia o modelo usando validação cruzada
        try:
            scores = cross_val_score(self.classifier, X_selected, self.y, 
                                     cv=self.cv, scoring='accuracy')
            cv_score = np.mean(scores)
            
            # Penalidade para muitas features - promove parcimônia
            # Alpha controla o peso da penalidade (ajuste conforme necessário)
            alpha = 0.001
            penalty = alpha * n_selected_features / self.n_features
            
            # Fitness final: acurácia - penalidade
            return cv_score - penalty
            
        except Exception as e:
            print(f"Erro ao avaliar fitness: {e}")
            return -np.inf
    
    def evaluate_population(self, population):
        """Avalia todos os indivíduos da população e retorna seus valores de fitness."""
        fitness = np.array([self.fitness_function(chromosome) for chromosome in population])
        return fitness
    
    def tournament_selection(self, population, fitness):
        """Seleciona um indivíduo da população usando seleção por torneio."""
        # Escolhe aleatoriamente indivíduos para o torneio
        tournament_indices = np.random.choice(len(population), TOURNAMENT_SIZE, replace=False)
        tournament_fitness = fitness[tournament_indices]
        
        # Retorna o melhor indivíduo do torneio
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index].copy()
    
    def crossover(self, parent1, parent2):
        """Realiza cruzamento usando dois pontos de corte."""
        if random.random() < CROSSOVER_RATE:
            # Seleciona dois pontos de corte
            points = sorted(random.sample(range(1, self.n_features), 2))
            
            # Cria dois filhos através do cruzamento de dois pontos
            child1 = np.concatenate([parent1[:points[0]], 
                                    parent2[points[0]:points[1]], 
                                    parent1[points[1]:]])
            
            child2 = np.concatenate([parent2[:points[0]], 
                                    parent1[points[0]:points[1]], 
                                    parent2[points[1]:]])
            
            return child1, child2
        else:
            # Se não houver cruzamento, retorna cópias dos pais
            return parent1.copy(), parent2.copy()
    
    def mutate(self, chromosome):
        """Aplica mutação bit a bit com uma determinada probabilidade."""
        for i in range(len(chromosome)):
            if random.random() < MUTATION_RATE:
                # Inverte o bit (0 para 1 ou 1 para 0)
                chromosome[i] = 1 - chromosome[i]
                
        # Garante que pelo menos MIN_FEATURES estejam selecionadas
        if np.sum(chromosome) < MIN_FEATURES:
            # Seleciona aleatoriamente algumas features para ativar
            zero_indices = np.where(chromosome == 0)[0]
            if len(zero_indices) > 0:
                to_activate = random.sample(list(zero_indices), 
                                           min(MIN_FEATURES - int(np.sum(chromosome)), len(zero_indices)))
                chromosome[to_activate] = 1
            
        return chromosome
    
    def ant_solution(self):
        """
        Constrói uma solução usando o algoritmo ACO.
        Cada formiga decide probabilisticamente quais features selecionar
        com base no feromônio e na informação heurística.
        """
        solution = np.zeros(self.n_features, dtype=np.int8)
        
        # Para cada feature, decide se deve selecioná-la
        for i in range(self.n_features):
            # Probabilidade baseada no feromônio e na heurística
            prob = (self.pheromone[i]**ALPHA) * (self.heuristic[i]**BETA)
            
            # Normaliza para um valor entre 0 e 1
            threshold = prob / (1.0 + prob)
            
            # Decisão probabilística
            if random.random() < threshold:
                solution[i] = 1
        
        # Garante o número mínimo de features
        if np.sum(solution) < MIN_FEATURES:
            zero_indices = np.where(solution == 0)[0]
            if len(zero_indices) > 0:
                to_activate = random.sample(list(zero_indices), 
                                           min(MIN_FEATURES - int(np.sum(solution)), len(zero_indices)))
                solution[to_activate] = 1
        
        return solution
    
    def update_pheromone(self, solutions, fitness_values):
        """
        Atualiza o feromônio com base nas soluções encontradas e seus fitness.
        """
        # Evaporação do feromônio
        self.pheromone *= (1 - EVAPORATION_RATE)
        
        # Depósito de feromônio pelas formigas
        for solution, fitness in zip(solutions, fitness_values):
            if fitness > 0:  # Apenas soluções com fitness positivo
                # Quanto melhor o fitness, maior o depósito
                deposit = Q * fitness
                
                # Deposita nas features selecionadas
                selected_indices = np.where(solution == 1)[0]
                self.pheromone[selected_indices] += deposit
        
        # Assegura valores mínimos de feromônio para evitar estagnação
        self.pheromone = np.maximum(self.pheromone, 0.1)
    
    def create_new_generation(self, population, fitness):
        """Cria uma nova geração usando GA e ACO."""
        new_population = np.zeros_like(population)
        
        # Elitismo: mantém o melhor indivíduo
        elite_index = np.argmax(fitness)
        new_population[0] = population[elite_index].copy()
        
        # Define quantos indivíduos usarão ACO versus GA
        aco_count = int(POPULATION_SIZE * ACO_RATIO)
        ga_count = POPULATION_SIZE - aco_count - 1  # -1 por causa do elite
        
        # Gera soluções usando ACO
        aco_solutions = []
        for i in range(aco_count):
            aco_solutions.append(self.ant_solution())
        
        # Avalia as soluções ACO
        aco_fitness = [self.fitness_function(sol) for sol in aco_solutions]
        
        # Atualiza o feromônio
        self.update_pheromone(aco_solutions, aco_fitness)
        
        # Adiciona as soluções ACO à nova população
        for i in range(aco_count):
            new_population[i+1] = aco_solutions[i]
        
        # Cria o restante da população usando GA
        for i in range(1 + aco_count, POPULATION_SIZE, 2):
            # Seleciona dois pais
            parent1 = self.tournament_selection(population, fitness)
            parent2 = self.tournament_selection(population, fitness)
            
            # Cruzamento
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutação
            child1 = self.mutate(child1)
            new_population[i] = child1
            
            if i + 1 < POPULATION_SIZE:
                child2 = self.mutate(child2)
                new_population[i + 1] = child2
        
        return new_population
    
    def run(self):
        """Executa o algoritmo híbrido GA-ACO."""
        # Inicializa a população
        population = self.initialize_population()
        
        # Loop principal
        for generation in range(MAX_GENERATIONS):
            # Avalia a população atual
            fitness = self.evaluate_population(population)
            
            # Atualiza a melhor solução global
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > self.g_best_fitness:
                self.g_best = population[current_best_idx].copy()
                self.g_best_fitness = fitness[current_best_idx]
            
            # Estatísticas da geração atual
            best_fitness = np.max(fitness)
            avg_fitness = np.mean(fitness)
            best_chromosome = population[np.argmax(fitness)]
            feature_count = np.sum(best_chromosome)
            
            # Armazena o histórico
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.feature_count_history.append(feature_count)
            
            # Exibe informações
            if generation % 5 == 0:
                print(f"Geração {generation}: Melhor fitness = {best_fitness:.4f}, " 
                      f"Features selecionadas = {feature_count}/{self.n_features}")
            
            # Cria a nova geração
            population = self.create_new_generation(population, fitness)
        
        # Resultado final
        print(f"\nResultado após {MAX_GENERATIONS} gerações:")
        print(f"Melhor fitness: {self.g_best_fitness:.6f}")
        print(f"Número de features selecionadas: {np.sum(self.g_best)}/{self.n_features}")
        
        # Índices das features selecionadas
        selected_features = np.where(self.g_best == 1)[0]
        print(f"Índices das features selecionadas: {selected_features}")
        
        # Visualiza resultados
        self.plot_results()
        
        return self.g_best, self.g_best_fitness, selected_features
    
    def plot_results(self):
        """Visualiza o progresso da otimização."""
        generations = range(len(self.best_fitness_history))
        
        # Plot do fitness
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(generations, self.best_fitness_history, 'b-', label='Melhor Fitness')
        plt.plot(generations, self.avg_fitness_history, 'r--', label='Fitness Médio')
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.title('Evolução do Fitness')
        plt.legend()
        plt.grid(True)
        
        # Plot do número de features
        plt.subplot(1, 2, 2)
        plt.plot(generations, self.feature_count_history, 'g-')
        plt.xlabel('Geração')
        plt.ylabel('Número de Features')
        plt.title('Número de Features Selecionadas')
        plt.axhline(y=self.n_features, color='r', linestyle='--', 
                   label=f'Total de Features ({self.n_features})')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()