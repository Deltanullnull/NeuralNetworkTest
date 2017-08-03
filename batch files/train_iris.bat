@echo off
call NeuralNetworkTest.exe train iris.data model model.bin lambda 0.00001 epsilon 0.0001
pause