library(rpart.plot)
library(mlr)
library(pROC)
library(caret)
library(rpart)
#przygotowanie danych
dane <- read.csv("dane_oczyszczone.csv", sep=";")
sezon2023 <- read.csv("dane4.csv", sep=";")
dane <- na.omit(dane)
sezon2023 <-na.omit(sezon2023)
sezon2023_s <- read.csv("sezon2023_s.csv", sep=";")
sezon2023_s <- na.omit(sezon2023_s)
sezon2023 <- sezon2023[,c(8,12,13,14,15,16,21,23,24,25,26,27)]
str(dane)
dane$HomeWin <- as.factor(dane$HomeWin)
dane_tree1 <- dane[,c(8,12,13,14,15,16,21,23,24,25,26,27)]
str(dane_tree1)
#Podział danych
set.seed(123)
ind <- sample(2, nrow(dane_tree1), replace = TRUE, prob = c(0.7, 0.3))
train_set <- dane_tree1[ind==1,]
test_set <- dane_tree1[ind==2,]



#Budowa podstawowego drzewa 
tree1 = rpart(HomeWin ~ ., 
              data=train_set, 
              method = 'class',
              #parms = list(split = "information")
              )

rpart.plot(tree1)

predicted <- predict(tree1, train_set, type = 'class')
confusionMatrix(predicted,train_set$HomeWin)

predicted1 <- predict(tree1, test_set, type = 'class')
confusionMatrix(predicted1,test_set$HomeWin)

probabilities <- predict(tree1, newdata = test_set, type = "prob")[,2]
roc_obj <- roc(test_set$HomeWin, probabilities)
auc <- round(auc(test_set$HomeWin, probabilities),4)



#Hyperparameter Tuning training with mlr
getParamSet("classif.rpart")
traintask <- makeClassifTask(
  data=train_set, 
  target="HomeWin"
)
# Define Grid
control_grid = makeTuneControlGrid()
# Define Cross Validation
resample = makeResampleDesc("CV",iter = 5,predict = "both")
# Define Measure
measure = acc

param_grid <- makeParamSet( 
  makeDiscreteParam("maxdepth", values=1:10),
  makeDiscreteParam("cp", values = 0),
  makeDiscreteParam("minsplit", values=1:30),
  makeDiscreteParam('xval',value =10)
)

dt_tuneparam_multi <- tuneParams(learner='classif.rpart', 
                                 task=traintask, 
                                 resampling = resample,
                                 measures = measure,
                                 par.set=param_grid, 
                                 control=control_grid, 
                                 show.info = TRUE)

# Extracting best Parameters from Multi Search
best_params = setHyperPars( 
  makeLearner("classif.rpart", predict.type = "prob"), 
  par.vals = dt_tuneparam_multi$x
)

best_model_multi <- mlr::train(best_params, traintask)
best_tree_model <- best_model_multi$learner.model
rpart.plot(best_tree_model)
predicted2 <- predict(best_tree_model, newdata = test_set, type = "class")
confusionMatrix(predicted2, test_set$HomeWin)

probabilities <- predict(best_tree_model, newdata = test_set, type = "prob")[,2]
roc_obj <- roc(test_set$HomeWin, probabilities)
auc <- round(auc(test_set$HomeWin, probabilities),4)


#przycinanie 
best_tree_model$cptable
best_cp <- best_tree_model$cptable[which.min(best_tree_model$cptable[,"xerror"]),"CP"]
plotcp(best_tree_model)
# Teraz możemy przyciąć nasze drzewo
pruned_tree <- prune(best_tree_model, cp = best_cp)
print(pruned_tree)

rpart.plot(pruned_tree,roundint = FALSE)
# Testowanie przyciętego drzewa na zbiorze testowym
predicted3 <- predict(pruned_tree, newdata = test_set, type = "class")
confusionMatrix(predicted3, test_set$HomeWin)


#sprawdzenie ktora zmienna ma najwieksze znaczenie 
importance <- varImp(pruned_tree)
print(importance)



#do wykresu
probabilities <- predict(pruned_tree, newdata = test_set, type = "prob")[,2]
roc_obj <- roc(test_set$HomeWin, probabilities)
auc <- round(auc(test_set$HomeWin, probabilities),4)

# Rysowanie krzywej ROC
ggroc(roc_obj,colour = 'steelblue',size=2,legacy.axes = TRUE) +
  geom_abline(linetype = "dashed") +
  theme(panel.border = element_rect(color = 'black',fill = NA,size = 1),
        panel.background = element_rect(fill='gray95'),
        plot.background = element_rect(color = 'black', size = 1) 
        )+
  ggtitle(paste0('Krzywa ROC ', '(AUC = ', auc, ')')) +
  labs(x = "1 - Swoistość",
       y = "Swoistość")



#stopa zwrotu
przewidywania <- predict(pruned_tree,sezon2023,type = 'class')
sezon2023_s$predykcje <- przewidywania
sezon2023_s$HomeWin <- as.numeric(sezon2023_s$HomeWin)
sezon2023_s$predykcje <- as.numeric(sezon2023_s$predykcje) - 1
str(sezon2023_s)
# Obliczamy sume zwrotu
sezon2023_s$return_rate <- ifelse(sezon2023_s$predykcje == sezon2023_s$HomeWin,
                                  ifelse(sezon2023_s$predykcje == 1,100*sezon2023_s$B365_1,100*sezon2023_s$B365_X2)-100,-100)
# Obliczamy łączną stopę zwrotu
total_return <- sum(sezon2023_s$return_rate)
total_return

