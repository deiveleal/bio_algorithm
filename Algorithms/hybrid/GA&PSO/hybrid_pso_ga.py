# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:00:00 2023
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class Hybrid_PSO_GA:
    def __init__(
            self,
            X,
            y,
            n_individuals=40,
            n_features=None,
            ga_ratio=0.6,
            pso_local_iterations=5,
            max_generations=50,
            crossover_prob=0.8,
            mutation_prob=0.1,
            w=0.7,
            c1=1.5,
            c2=1.5,
            classifier=None,
            cv=5,
            alpha=0.01,
            min_features=2,
            feature_names=None
    ):
        """
        Inicializa o algoritmo híbrido PSO-GA para seleção de características.

        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dados de entrada
        y : array-like, shape (n_samples,)
            Classes alvo
        n_individuals : int
            Tamanho da população/enxame
        n_features : int, opcional
            Número de características a considerar (padrão: todas)
        ga_ratio : float (0-1)
            Proporção da população inicializada usando GA puro (aleatório)
        pso_local_iterations : int
            Número de iterações PSO para refinamento local
        max_generations : int
            Número máximo de gerações
        crossover_prob : float
            Probabilidade de crossover
        mutation_prob : float
            Probabilidade de mutação
        w, c1, c2 : float
            Parâmetros do PSO (inércia, componente cognitivo,
            componente social)
        classifier : objeto classificador, opcional
        cv : int
            Número de folds para validação cruzada
        alpha : float
            Peso do termo de regularização para penalizar muitas
            características
        min_features : int
            Número mínimo de características a serem selecionadas
        """
        # Normalizar os dados de entrada
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X)
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.n_features = X.shape[1] if n_features is None else n_features
        self.n_individuals = n_individuals
        self.ga_ratio = ga_ratio
        self.pso_local_iterations = pso_local_iterations
        self.max_generations = max_generations
        self.min_features = min_features
        self.feature_names = feature_names

        # Parâmetros GA
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # Parâmetros PSO
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Classificador
        if classifier is None:
            self.classifier = KNeighborsClassifier(n_neighbors=3)
        else:
            self.classifier = classifier
        self.cv = cv

        # Penalidade para regularização
        self.alpha = alpha

        # Proporções de população
        self.n_ga = int(n_individuals * ga_ratio)
        self.n_pso = n_individuals - self.n_ga

        # Métricas para acompanhamento
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.feature_count_history = []
        self.execution_times = []
        self.best_solution = None
        # Melhor fitness inicial (menor valor possível)
        self.best_fitness = -np.inf
        self.start_time = None

    def fitness(self, solution):
        """Avalia o fitness de uma solução (subconjunto de características)."""
        # Se nenhuma característica for selecionada, retorne pontuação mínima
        n_selected = np.sum(solution)
        if n_selected < self.min_features:
            return 0

        # Selecione as características ativas
        selected_features = np.where(solution == 1)[0]
        # Seleciona todas as linhas e apenas as colunas selecionadas
        X_selected = self.X[:, selected_features]

        # Avalie usando validação cruzada
        try:
            scores = cross_val_score(
                self.classifier,
                X_selected,
                self.y,
                cv=self.cv,
                scoring='accuracy'
                )
            accuracy = np.mean(scores)

            # Penalize pelo número de características (termo de regularização)
            # Normalizado para estar entre 0 e 1
            penalty = self.alpha * (n_selected / self.n_features)

            return accuracy - penalty
        except Exception as e:
            # Se ocorrer um erro na validação cruzada
            # (por exemplo, uma classe ter poucos exemplos),
            # retorna um valor baixo de fitness
            print(f"Erro na validação cruzada: {e}")
            return 0

    def initialize_population(self):
        """
        Inicializa a população usando GA (aleatório) e PSO.
        Garante que pelo menos um indivíduo contenha todas as características.
        """
        # Primeiro indivíduo: todas as características selecionadas
        all_features_individual = np.ones(self.n_features, dtype=int)

        # Segundo indivíduo: solução com características selecionadas
        # aleatoriamente mas garantindo o mínimo de características
        random_individual = np.zeros(self.n_features, dtype=int)
        # Seleciona no mínimo min_features
        n_selected = max(self.min_features, int(0.3 * self.n_features))
        selected_features = np.random.choice(
            self.n_features,
            n_selected,
            replace=False
            )
        random_individual[selected_features] = 1

        # Inicializar parte da população aleatoriamente (GA)
        ga_population = np.zeros((self.n_ga - 2, self.n_features), dtype=int)

        for i in range(self.n_ga - 2):
            # Seleciona aleatoriamente algumas features
            # (pelo menos min_features)
            n_selected = np.random.randint(self.min_features, self.n_features)
            # Esta linha está selecionando aleatoriamente n_selected índices
            # únicos de características do conjunto total de self.n_features
            # características. Por exemplo, se self.n_features for 100 e
            # n_selected for 10, a função retornará um array com 10 índices
            # diferentes, cada um entre 0 e 99, representando as
            # características escolhidas aleatoriamente.
            selected_features = np.random.choice(
                self.n_features, n_selected, replace=False)
            ga_population[i, selected_features] = 1

        # Combinar os indivíduos especiais com o resto da população GA
        ga_population = np.vstack(
            [all_features_individual, random_individual, ga_population])

        print(f"\nPopulação GA inicializada com {self.n_ga} indivíduos.")
        print(ga_population)

        # Inicializar o restante usando PSO
        pso_positions = np.zeros((self.n_pso, self.n_features), dtype=int)
        pso_velocities = np.random.uniform(-1, 1, (self.n_pso, self.n_features))

        # Inicializar posições PSO aleatoriamente
        for i in range(self.n_pso):
            n_selected = np.random.randint(self.min_features, self.n_features)
            selected_features = np.random.choice(
                self.n_features, n_selected, replace=False)
            pso_positions[i, selected_features] = 1

        # Avaliar as posições iniciais
        pso_fitnesses = np.array([self.fitness(pos) for pos in pso_positions])

        # Definir melhores pessoais
        pso_best_positions = pso_positions.copy()
        pso_best_fitnesses = pso_fitnesses.copy()

        # Encontrar o melhor global
        best_idx = np.argmax(pso_fitnesses)
        global_best_position = pso_positions[best_idx].copy()

        # Executar algumas iterações de PSO para melhorar essa parte da população inicial
        for _ in range(3):  # Poucas iterações para inicialização
            for i in range(self.n_pso):
                # Atualizar velocidade
                r1, r2 = np.random.random(2)
                cognitive = self.c1 * r1 * \
                    (pso_best_positions[i] - pso_positions[i])
                social = self.c2 * r2 * \
                    (global_best_position - pso_positions[i])
                pso_velocities[i] = self.w * \
                    pso_velocities[i] + cognitive + social

                # Limitar velocidade
                pso_velocities[i] = np.clip(pso_velocities[i], -4, 4)

                # Atualizar posição (para valores binários)
                prob = 1 / (1 + np.exp(-pso_velocities[i]))
                pso_positions[i] = (np.random.random(
                    self.n_features) < prob).astype(int)

                # Garantir número mínimo de características
                if np.sum(pso_positions[i]) < self.min_features:
                    zeros = np.where(pso_positions[i] == 0)[0]
                    to_flip = np.random.choice(
                        zeros,
                        self.min_features - np.sum(pso_positions[i]),
                        replace=False)
                    pso_positions[i, to_flip] = 1

                # Avaliar nova posição
                fitness = self.fitness(pso_positions[i])

                # Atualizar melhor pessoal se necessário
                if fitness > pso_best_fitnesses[i]:
                    pso_best_fitnesses[i] = fitness
                    pso_best_positions[i] = pso_positions[i].copy()

                    # Atualizar o melhor global se necessário
                    if fitness > self.fitness(global_best_position):
                        global_best_position = pso_positions[i].copy()

        print(f"\nPopulação PSO inicializada com {self.n_pso} indivíduos.")
        print(pso_positions)

        # Combinar as populações GA e PSO
        population = np.vstack([ga_population, pso_positions])
        print(f"\nPopulação inicializada com {self.n_ga + self.n_pso} indivíduos.")
        print(population)

        return population

    def selection(self, population, fitnesses):
        """Seleção por torneio."""
        selected_indices = []
        for _ in range(len(population)):
            # Selecionar três indivíduos aleatoriamente para o torneio
            contestants = np.random.choice(len(population), 3, replace=False)
            # O vencedor é o que tem maior fitness
            winner = contestants[np.argmax(fitnesses[contestants])]
            selected_indices.append(winner)

        # Garantir que population seja um array NumPy
        population_array = np.array(population)
        return population_array[selected_indices]

    def crossover(self, parent1, parent2):
        """Crossover uniforme."""
        if np.random.random() < self.crossover_prob:
            # Criar máscara de crossover
            mask = np.random.randint(0, 2, size=self.n_features)
            # Aplicar máscara
            child1 = parent1.copy()
            child2 = parent2.copy()
            child1[mask == 1] = parent2[mask == 1]
            child2[mask == 1] = parent1[mask == 1]
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        """Mutação bit a bit com taxa adaptativa."""
        # Taxa de mutação adaptativa baseada no número de características
        # selecionadas
        current_ratio = np.sum(individual) / self.n_features

        # Aumenta a taxa de mutação se muitas características estão
        # selecionadas
        adaptive_rate = self.mutation_prob
        if current_ratio > 0.7:  # Se mais de 70% das características estão selecionadas
            adaptive_rate = min(0.2, self.mutation_prob *
                                2)  # Aumenta a taxa, máximo 20%
        elif current_ratio < 0.1:  # Se menos de 10% das características estão selecionadas
            adaptive_rate = max(0.005, self.mutation_prob /
                                2)  # Diminui a taxa, mínimo 0.5%

        for i in range(self.n_features):
            if np.random.random() < adaptive_rate:
                individual[i] = 1 - individual[i]  # Inverter o bit

        # Garantir número mínimo de características
        if np.sum(individual) < self.min_features:
            zeros = np.where(individual == 0)[0]
            if len(zeros) > 0:
                to_flip = np.random.choice(
                    zeros,
                    min(
                        self.min_features - int(np.sum(individual)),
                        len(zeros)),
                    replace=False)
                individual[to_flip] = 1

        return individual

    def pso_local_search(self, individual, best_global):
        """
        Aplica PSO como operador de busca local.
        """
        position = individual.copy()
        velocity = np.random.uniform(-1, 1, self.n_features)
        personal_best = position.copy()
        personal_best_fitness = self.fitness(position)

        for i in range(self.pso_local_iterations):
            # Atualizar velocidade com inércia decrescente
            # Reduz inércia ao longo das iterações
            inertia = self.w * (1 - 0.5 * i / self.pso_local_iterations)
            r1, r2 = np.random.random(2)
            cognitive = self.c1 * r1 * (personal_best - position)
            social = self.c2 * r2 * (best_global - position)
            velocity = inertia * velocity + cognitive + social

            # Limitar velocidade
            velocity = np.clip(velocity, -4, 4)

            # Atualizar posição (para valores binários)
            prob = 1 / (1 + np.exp(-velocity))
            new_position = (np.random.random(
                self.n_features) < prob).astype(int)

            # Garantir número mínimo de características
            if np.sum(new_position) < self.min_features:
                zeros = np.where(new_position == 0)[0]
                if len(zeros) > 0:
                    to_flip = np.random.choice(zeros,
                                               min(self.min_features -
                                                   int(np.sum(new_position)), len(zeros)),
                                               replace=False)
                    new_position[to_flip] = 1

            # Avaliar nova posição
            new_fitness = self.fitness(new_position)

            # Atualizar melhor pessoal se necessário
            if new_fitness > personal_best_fitness:
                personal_best_fitness = new_fitness
                personal_best = new_position.copy()
                position = new_position.copy()
            else:
                # Aplicar uma versão de simulated annealing para evitar
                # mínimos locais
                temperature = 1.0 / (i + 1)  # Diminui com as iterações
                if np.random.random() < np.exp(
                    (new_fitness - personal_best_fitness) / temperature):
                    position = new_position.copy()

        return personal_best, personal_best_fitness

    def run(self):
        """Executa o algoritmo híbrido PSO-GA."""
        self.start_time = time.time()

        # Inicializar população (GA+PSO)
        population = self.initialize_population()

        # Avaliar população inicial
        fitnesses = np.array([self.fitness(ind) for ind in population])

        # Encontrar o melhor global inicial
        best_idx = np.argmax(fitnesses)
        global_best = population[best_idx].copy()
        global_best_fitness = fitnesses[best_idx]

        # Atualizar métricas
        self.best_fitness_history.append(global_best_fitness)
        self.avg_fitness_history.append(np.mean(fitnesses))
        self.feature_count_history.append(np.sum(global_best))
        self.execution_times.append(0.0)

        # Atualizar melhor solução
        if global_best_fitness > self.best_fitness:
            self.best_fitness = global_best_fitness
            self.best_solution = global_best.copy()

        print("\nGerações:")
        for generation in range(self.max_generations):
            gen_start_time = time.time()

            # Seleção
            selected_population = self.selection(population, fitnesses)

            # Criar nova população
            new_population = []

            # Elitismo: manter o melhor indivíduo
            new_population.append(global_best)

            # Aplicar operadores genéticos ao resto da população
            for i in range(0, len(population)-1, 2):
                if i+1 < len(population):
                    parent1, parent2 = selected_population[i], selected_population[i+1]
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    new_population.extend([child1, child2])
                else:
                    child = self.mutate(selected_population[i].copy())
                    new_population.append(child)

            # Garantir que o tamanho da população seja constante
            new_population = new_population[:self.n_individuals]

            # Aplicar PSO como busca local a uma fração da população
            # 20% da população ou pelo menos 3
            n_pso_local = max(3, int(0.2 * self.n_individuals))

            # Avaliar nova população
            fitnesses = np.array([self.fitness(ind) for ind in new_population])

            # Encontrar indivíduos para aplicar PSO local
            pso_candidates_idx = np.argsort(fitnesses)[-n_pso_local:]

            # Aplicar PSO local
            for idx in pso_candidates_idx:
                improved_solution, improved_fitness = self.pso_local_search(
                    new_population[idx], global_best)

                # Substituir a solução se melhorou
                if improved_fitness > fitnesses[idx]:
                    new_population[idx] = improved_solution
                    fitnesses[idx] = improved_fitness

                    # Atualizar o melhor global se necessário
                    if improved_fitness > global_best_fitness:
                        global_best = improved_solution.copy()
                        global_best_fitness = improved_fitness

            # Atualizar população
            population = new_population

            # Calcular tempo desta geração
            gen_time = time.time() - gen_start_time
            self.execution_times.append(gen_time)

            # Atualizar métricas
            self.best_fitness_history.append(global_best_fitness)
            self.avg_fitness_history.append(np.mean(fitnesses))
            self.feature_count_history.append(np.sum(global_best))

            # Atualizar melhor solução
            if global_best_fitness > self.best_fitness:
                self.best_fitness = global_best_fitness
                self.best_solution = global_best.copy()

            # Imprimir progresso
            if (generation + 1) % 5 == 0 or generation == 0:
                n_selected = np.sum(global_best)
                total_time = time.time() - self.start_time
                print(f"Geração {generation+1}/{self.max_generations}: Melhor fitness = {global_best_fitness:.4f}, "
                      f"Média fitness = {np.mean(fitnesses):.4f}, Características = {n_selected}/{self.n_features}, "
                      f"Tempo = {total_time:.2f}s")

        # Resultado final
        print(f"\nResultado após {self.max_generations} gerações:")

        print(f"Melhor fitness: {global_best_fitness:.6f}")
        print(
            f"Número de features selecionadas: {np.sum(global_best)}/{self.n_features}")

        # Índices das features selecionadas
        selected_features = np.where(global_best == 1)[0]
        print(f"Índices das features selecionadas: {selected_features}")

        # Comparativo de acurácia entre usar todas as features e só as selecionadas
        print("\n=== Comparativo de Acurácia ===")
        # Avalia modelo com todas as features
        all_features_scores = cross_val_score(
            self.classifier, self.X, self.y, cv=self.cv, scoring='accuracy')
        all_features_accuracy = np.mean(all_features_scores)
        all_features_std = np.std(all_features_scores)

        # Avalia modelo só com features selecionadas
        X_selected = self.X[:, selected_features]
        selected_scores = cross_val_score(
            self.classifier, X_selected, self.y, cv=self.cv, scoring='accuracy')
        selected_accuracy = np.mean(selected_scores)
        selected_std = np.std(selected_scores)

        # Imprime resultados
        print(
            f"Todas as features ({self.n_features}): {all_features_accuracy:.6f} ± {all_features_std:.6f}")
        print(
            f"Features selecionadas ({len(selected_features)}): {selected_accuracy:.6f} ± {selected_std:.6f}")
        print(f"Diferença: {selected_accuracy - all_features_accuracy:.6f}")
        print(
            f"Redução de complexidade: {(1 - len(selected_features)/self.n_features)*100:.2f}%")

        # Retornar a melhor solução encontrada
        selected_features = np.where(self.best_solution == 1)[0]
        total_time = time.time() - self.start_time
        print(f"Tempo total de execução: {total_time:.2f} segundos")

        print("Nomes das características selecionadas:")
        for name in selected_features:
            print(f" - {self.feature_names[name]}")

        return selected_features, self.best_fitness, self.best_solution

    def plot_convergence(self):
        # Configuração da figura
        plt.figure(figsize=(8.27, 11.69))  # Tamanho A4

        # Plot de convergência na parte superior
        plt.subplot(2, 1, 1)
        generations = range(len(self.best_fitness_history))
        plt.plot(generations, self.best_fitness_history, 'b-', linewidth=2, label='Melhor Fitness')
        plt.plot(generations, self.avg_fitness_history, 'r--', linewidth=1, label='Fitness Médio')
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.title('Convergência do Fitness')
        plt.legend()
        plt.grid(True)

        # Tabela com resultados na parte inferior
        plt.subplot(2, 1, 2)
        plt.axis('off')  # Desativa os eixos

        # Título específico para a seção da tabela, posicionado corretamente
        plt.text(0.5, 0.95, 'Resultados de Execução', 
                 fontsize=14, ha='center', weight='bold', 
                 transform=plt.gca().transAxes)

        # Criar tabela com os dados
        table_data = [
            ['Parâmetro', 'Valor'],
            ['Total de características', f"{self.n_features}"],
            ['Tamanho da população', f"{self.n_individuals}"],
            ['Gerações', f"{self.max_generations}"],
            ['GA ratio', f"{self.ga_ratio}"],
            ['PSO local iterations', f"{self.pso_local_iterations}"],
            ['Crossover probability', f"{self.crossover_prob}"],
            ['Mutation probability', f"{self.mutation_prob}"],
            ['Best fitness', f"{self.best_fitness:.6f}"],
            ['Features selecionadas',
                f"{np.sum(self.best_solution)}/{self.n_features}"]
        ]

        table = plt.table(cellText=table_data, loc='center', cellLoc='center',
                          colWidths=[0.3, 0.3], bbox=[0.2, 0.1, 0.6, 0.7])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # Ajusta altura da tabela

        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
        plt.savefig('convergencia_A4.pdf', format='pdf', dpi=300)
        plt.show()

    def plot_results(self):
        """Visualiza o progresso da otimização em formato A4."""
        generations = range(len(self.best_fitness_history))

        # Tamanho A4 em polegadas
        plt.figure(figsize=(8.27, 11.69))

        # Plot do número de features
        plt.subplot(2, 1, 1)
        plt.plot(generations, self.feature_count_history, 'g-', linewidth=2, 
                label=f'Features Selecionadas')
        plt.xlabel('Geração')
        plt.ylabel('Número de Features')
        plt.title('Número de Features Selecionadas')
        plt.axhline(y=self.n_features, color='r', linestyle='--',
                    label=f'Total de Features ({self.n_features})')
        
        # Obtém o valor final de features selecionadas para a legenda
        final_selected = self.feature_count_history[-1]
        
        # Configurar o eixo Y para mostrar apenas valores inteiros
        ax = plt.gca()
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Ajustar os limites do eixo Y para ter margem adequada
        # Encontrar o valor mínimo e máximo no histórico
        min_features = min(self.feature_count_history)
        max_features = max(max(self.feature_count_history), self.n_features)

        # Adicionar margem de 1 unidade abaixo e acima
        plt.ylim(max(1, min_features - 1), max_features + 1)

        # Atualiza a legenda para mostrar o número final de features selecionadas
        plt.legend([f'Features Selecionadas ({final_selected})', 
                    f'Total de Features ({self.n_features})'])
        
        plt.grid(False)
        plt.tight_layout(pad=2.0)
        plt.savefig('resultados_A4.pdf', format='pdf', dpi=300)
        plt.show()