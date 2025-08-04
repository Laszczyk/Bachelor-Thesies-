library(ranger)
library(ggplot2)
library(caret)
library(pROC)
#wczytanie danych 
dane <- read.csv("C:/Users/kuba/Desktop/praca licencjacka/dane3.csv", sep=";")
dane <-na.omit(dane)
sezon2023 <- read.csv("C:/Users/kuba/OneDrive/Pulpit/praca licencjacka/dane4.csv", sep=";")
dane <- na.omit(dane)
sezon2023 <-na.omit(sezon2023)
sezon2023_s <- read.csv("C:/Users/kuba/OneDrive/Pulpit/praca licencjacka/sezon2023_s.csv", sep=";")
sezon2023_s <- na.omit(sezon2023_s)
sezon2023 <- sezon2023[,c(2,3,8,10,11,12,13,14,15,16,21,23,24,25,26,27)]


str(dane)


dane$HomeWin  <- as.factor(dane$HomeWin)
daneForest <- dane[,c(2,3,8,10,11,12,13,14,15,16,21,23,24,25,26,27)] 


#Podział danych 
set.seed(123)
ind <- sample(2, nrow(daneForest), replace = TRUE, prob = c(0.7, 0.3))
train_set <- daneForest[ind==1,]
test_set <- daneForest[ind==2,]


#podstawowy model 
model <- ranger(HomeWin ~ ., 
                       data = train_set 
                )

#predykcja i confusion matrix - train data
p1 <- predict(model, train_set)
confusionMatrix(p1$predictions, train_set$HomeWin)
# predykcja i confusion matrix  - test data
p2 <- predict(model, test_set)
confusionMatrix(p2$predictions, test_set$HomeWin)

#wybor num.trees
num.trees <- seq(50, 1000, by = 50)
oob.error <- vector("numeric", length(num.trees))

#mtry ustawiane jako pierwiastek z ilosci zmiennych 
# Iterujemy przez różne liczby drzew
for (i in seq_along(num.trees)) {
  model <- ranger(HomeWin ~ ., 
                  data = train_set, 
                  num.trees = num.trees[i], 
                  mtry = 4, 
                  importance = 'impurity')
  oob.error[i] <- model$prediction.error
}

# Wizualizujemy wyniki
df <- data.frame(NumTrees = num.trees, OOBE = oob.error)
ggplot(df, aes(x = NumTrees, y = OOBE)) +
  geom_line() +
  labs(title = "Błąd prognozy w zależnośći od liczby drzew", x = "Liczba drzew", y = "Błąd prognozy")+
  theme(panel.border = element_rect(color = 'black',fill = NA,size = 1),
        panel.background = element_rect(fill='gray95'),
        plot.background = element_rect(color = 'black', size = 1)
        )

min.error.index <- which.min(oob.error)
best.num.trees <- num.trees[min.error.index]


# Stworzenie siatki parametrów do przetestowania
tune_grid <- expand.grid(max.depth = seq(1, 15, by = 1),
                         mtry.size = seq(1, 10, by = 1))

# Iteracja po siatce parametrów
for(i in 1:nrow(tune_grid)){
  max_depth_value <- tune_grid$max.depth[i]
  mtry_value <- tune_grid$mtry.size[i]
  
  model <- ranger(HomeWin ~ ., 
                  data = train_set, 
                  num.trees = best.num.trees, 
                  mtry = mtry_value, 
                  max.depth = max_depth_value,
                  importance = 'impurity')
  oob.error[i] <- model$prediction.error
}

min.error.index <- which.min(oob.error)
best_params <- tune_grid[min.error.index,]

set.seed(123)
#finalny model i predykcja 
model <- ranger(HomeWin ~ ., 
                data = train_set, 
                num.trees = best.num.trees, 
                mtry = best_params$mtry.size,
                max.depth = best_params$max.depth,
                importance = 'impurity')


print(model)
attributes(model)
#predykcja i confusion matrix - train data
p3 <- predict(model, train_set)
confusionMatrix(p3$predictions, train_set$HomeWin)
# predykcja i confusion matrix  - test data
p4 <- predict(model, test_set)
confusionMatrix(p4$predictions, test_set$HomeWin)


num_iterations <- 10  # Liczba iteracji w pętli
mean_accuracy <- 0

for (i in 1:num_iterations) {
  # Trenowanie modelu Random Forest
  model <- ranger(HomeWin ~ ., 
                  data = train_set, 
                  num.trees = best.num.trees, 
                  mtry = best_params$mtry.size, 
                  max.depth = best_params$max.depth,
                  importance = 'impurity')
  
  # Przewidywanie na danych testowych
  przewidywania <- predict(model, data = test_set)$predictions
  
  # Obliczanie accuracy dla aktualnej iteracji
  accuracy <- sum(przewidywania == test_set$HomeWin) / length(przewidywania)
  
  # Dodawanie accuracy do sumy
  mean_accuracy <- mean_accuracy + accuracy
}

# Obliczanie średniej wartości accuracy
mean_accuracy <- mean_accuracy / num_iterations

# Wyświetlanie wyniku
cat("Mean accuracy:", mean_accuracy, "\n")



#stopa zwrotu
przewidywania <- predict(model,sezon2023)
sezon2023_s$predykcje <- przewidywania$predictions
# Obliczamy stopę zwrotu
sezon2023_s$return_rate <- ifelse(sezon2023_s$predykcje == sezon2023_s$HomeWin,
                                  ifelse(sezon2023_s$predykcje == 1,100*sezon2023_s$B365_1,100*sezon2023_s$B365_X2)-100,-100)
# Obliczamy łączną stopę zwrotu
total_return <- sum(sezon2023_s$return_rate)
total_return


#model do wykresu 
model <- ranger(HomeWin ~ ., 
                data = train_set, 
                num.trees = 700, 
                mtry = 3, 
                min.node.size = 17, 
                max.depth = 4,
                probability = TRUE,
                importance = 'impurity')
#wykres
predictions <- predict(model, data = test_set)
probabilities <- predictions$predictions
roc_obj <- roc(test_set$HomeWin, probabilities)
auc <- round(auc(test_set$HomeWin, probabilities),4)
# Rysowanie krzywej ROC
ggroc(roc_obj,colour = 'steelblue',size=2,legacy.axes = TRUE) +
  geom_abline(linetype = "dashed") +
  theme(panel.border = element_rect(color = 'black',fill = NA,size = 1),
        panel.background = element_rect(fill='gray95'),
        plot.background = element_rect(color = 'black', size = 1), 
        axis.title.x = element_blank(),  
        axis.title.y = element_blank())+
  ggtitle(paste0('ROC Curve ', '(AUC = ', auc, ')')) +
  labs(x = "1 - Specificity",
       y = "Sensitivity")

#waga zmiennych 
importance <- model$variable.importance
write.csv(importance, file = "importance.csv", row.names = TRUE)
