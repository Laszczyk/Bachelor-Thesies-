# Packages
library(xgboost)
library(Matrix)
library(caret)
library(mlr)
library(pROC)
# Data
dane <- read.csv("C:/Users/kuba/OneDrive/Pulpit/praca licencjacka/dane_oczyszczone.csv", sep=";")
#dane1 <- read.csv("C:/Users/kuba/OneDrive/Pulpit/praca licencjacka/razdwa.csv", sep=";")
#dane1 <- na.omit(dane1)
#dane <-na.omit(dane)

dane$HomeWin  <- as.numeric(dane$HomeWin)
#dane podstawowe
danexboost <- dane[,c(2,3,8,11,12,13,14,15,16,21,23,24,25,26,27)] 


#Podział danych
set.seed(123)
ind <- sample(2, nrow(danexboost), replace = TRUE, prob = c(0.7, 0.3))
train_set <- danexboost[ind==1,]
test_set <- danexboost[ind==2,]

xgtrain <- xgb.DMatrix(data = as.matrix(train_set[, -length(test_set)]), label = train_set[,length(test_set)])
xgtest <- xgb.DMatrix(data = as.matrix(test_set[, -length(test_set)]), label = test_set[,length(test_set)])

# Parameters
xgb_params <- list("objective" = "binary:logistic",
                   "eval_metric" = "logloss"
                   )
watchlist <- list(train = xgtrain, test = xgtest)

xg_model <- xgb.train(params = xgb_params,
                      data = xgtrain,
                      booster = "gbtree",
                      nrounds = 413,
                      watchlist = watchlist,
                      eta = 0.01,
                      max.depth =4,
                      gamma = 0,
                      subsample = 0.842,
                      colsample_bytree = 0.764,
                      #missing = NA,
                      verbose = 2,
                      set.seed(123))


logframe <- data.frame(xg_model$evaluation_log)
plot(logframe$iter, logframe$train_logloss, col = 'blue')
lines(logframe$iter, logframe$test_logloss, col = 'red')

min(logframe$test_logloss)
logframe[logframe$test_logloss == min(logframe$test_logloss),]

params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.03, 
               gamma=0, max_depth=3, min_child_weight=4.98, subsample=0.704, colsample_bytree=0.992)

xgbcv <- xgb.cv( params = params, data = xgtrain, nrounds = 230, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, early_stop_round = 20, maximize = F)


xgbcv$evaluation_log[xgbcv$evaluation_log$test_logloss_mean==min(xgbcv$evaluation_log$test_logloss_mean),]

# predykcja i confusion matrix - test data
preds <- predict(xg_model, xgtest)
preds <- ifelse(preds > 0.5, 1, 0)
preds <- as.factor(preds)
x <- as.factor(test_set[,length(test_set)])
# Ocena modelu
table(pred = preds, true = test_set[,length(test_set)]) 
confusionMatrix(preds,x)

 # predykcja i confusion matrix - train data
preds <- predict(xg_model, xgtrain)
preds <- ifelse(preds > 0.5, 1, 0)
preds <- as.factor(preds)
x <- as.factor(train_set[,length(train_set)])
# Ocena modelu
table(pred = preds, true = train_set[,length(train_set)]) 
confusionMatrix(preds,x)




#krzywa roc
prob_preds <- predict(xg_model, xgtest)
roc_obj <- roc(test_set$HomeWin, prob_preds)
auc <- round(auc(test_set$HomeWin, prob_preds),4)

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





# Feature importance
imp <- xgb.importance(colnames(xgtrain), model = xg_model)
xgb.plot.importance(imp)
train_set$HomeWin  <- as.factor(train_set$HomeWin)



test_set$HomeWin <- as.factor(test_set$HomeWin)
train_set$HomeWin <- as.factor(train_set$HomeWin)
traintask <- makeClassifTask (data = train_set,target = "HomeWin")
testtask <- makeClassifTask (data = test_set,target = "HomeWin")

lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=29, eta=0.1,set.seed(123))


params <- makeParamSet(makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                       makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                       makeNumericParam("subsample",lower = 0.5,upper = 1), 
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))


resample <- makeResampleDesc("CV",stratify = T,iters=5L)
ctrl <- makeTuneControlRandom(maxit = 10L)
tune_params <- tuneParams(learner = lrn, task = traintask, 
                     resampling = resample, measures = acc, 
                     par.set = params ,control = ctrl,show.info = T)

tune_params$y 

#set hyperparameters

lrn_tune <- setHyperPars(lrn, par.vals = tune_params$x)
print(tune_params$x)
#train model
xgmodel <- mlr::train(learner=lrn_tune, task=traintask)

#predykcja
xgpred <- predict(xgmodel,testtask)
confusionMatrix(xgpred$data$response,xgpred$data$truth)

















#stopa zwrotu
sezon2023 <- read.csv("C:/Users/kuba/OneDrive/Pulpit/praca licencjacka/dane4.csv", sep=";")
dane <- na.omit(dane)
sezon2023 <-na.omit(sezon2023)
sezon2023_s <- read.csv("C:/Users/kuba/OneDrive/Pulpit/praca licencjacka/sezon2023_s.csv", sep=";")
sezon2023_s <- na.omit(sezon2023_s)
sezon2023 <- sezon2023[,c(2,3,8,11,12,13,14,15,16,21,23,24,25,26,27)]




#dla podstawowego modelu 
dsezon2023 <- xgb.DMatrix(data = as.matrix(sezon2023[, -length(sezon2023)]), label = sezon2023[,length(sezon2023)])
przewidywania <- predict(xg_model,dsezon2023)
przewidywania <- ifelse(przewidywania > 0.5, 1, 0)
przewidywania <- as.factor(przewidywania)
sezon2023_s$predykcje <- przewidywania 

# Obliczamy stopę zwrotu
sezon2023_s$return_rate <- ifelse(sezon2023_s$predykcje == sezon2023_s$HomeWin,
                                  ifelse(sezon2023_s$predykcje == 1,100*sezon2023_s$B365_1,100*sezon2023_s$B365_X2)-100,-100)
# Obliczamy łączną stopę zwrotu
total_return <- sum(sezon2023_s$return_rate)
total_return


#dla finalnego modelu 
testtask <- makeClassifTask (data = sezon2023,target = "HomeWin")
xgpred <- predict(xgmodel,testtask)
sezon2023_s$prawdopodobienstwo <- przewidywania 

#stopa zwrotu
sezon2023_s$predykcje <- xgpred$data$response 

# Obliczamy stopę zwrotu
sezon2023_s$return_rate <- ifelse(sezon2023_s$predykcje == sezon2023_s$HomeWin,
                                  ifelse(sezon2023_s$predykcje == 1,100*sezon2023_s$B365_1,100*sezon2023_s$B365_X2)-100,-100)
# Obliczamy łączną stopę zwrotu
total_return <- sum(sezon2023_s$return_rate)
total_return





# Przygotowanie danych do predykcji
dsezon2023 <- xgb.DMatrix(data = as.matrix(sezon2023[, -length(sezon2023)]))
set.seed(123)
# Inicjalizacja wektora na łączne stopy zwrotu
total_returns <- numeric(0)

# Pętla wykonująca predykcję i obliczająca łączną stopę zwrotu 10 razy
for (i in 1:10) {
  # Trenowanie modelu
  xg_model <- xgb.train(params = xg_param,
                         data = xgtrain,
                         booster = "gbtree",
                         nrounds = 29,
                         watchlist = watchlist,
                         eta = 0.1,
                         max.depth =5,
                         #gamma = 0,
                         subsample = 0.8,
                         colsample_bytree = 1,
                         min_child_weight =1,
                         set.seed(123)
  )
  # Przewidywanie wyników
  przewidywania <- predict(xg_model, dsezon2023)
  przewidywania <- ifelse(przewidywania > 0.5, 1, 0)
  przewidywania <- as.factor(przewidywania)
  sezon2023_s$predykcje <- przewidywania
  # Obliczanie stopy zwrotu
  sezon2023_s$return_rate <- ifelse(sezon2023_s$predykcje == sezon2023_s$HomeWin,
                                    ifelse(sezon2023_s$predykcje == 1,100*sezon2023_s$B365_1,100*sezon2023_s$B365_X2)-100,-100)
  
  # Obliczanie łącznej stopy zwrotu
  total_return <- sum(sezon2023_s$return_rate)
  total_returns <- c(total_returns, total_return)
  
  cat("Iteracja", i, "- Łączna stopa zwrotu:", total_return, "\n")
}
print(mean(total_returns))

