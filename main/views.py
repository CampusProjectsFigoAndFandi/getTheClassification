from django.http import HttpResponse
from django.shortcuts import render
from .getCluster import getCluster

def home(request): 
    if request.method == 'GET':
        return render(request, "home.html")
    elif request.method == 'POST':
        answers = [] 
        for i in range(1,10):
            answers.append(request.POST.get(f'jawaban_{i}'))
        cluster,accuracy = getCluster(answers)
        if (cluster == 1 ):
            valueCluster = "Puas"
        elif (cluster == 0 ):
            valueCluster = "Tidak Puas"
        else:
            valueCluster = "Gagal Mengcluster"
        context = {'cluster': cluster,'valueCluster':valueCluster,'accuracy':str(round(accuracy*100,1))+'%'}
        return render(request,'output.html', context)
        # return HttpResponse(cluster)
