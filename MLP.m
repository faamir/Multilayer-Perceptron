%in barname baraye amuzeshe shabake asabi baraye amalgar zarb dar shabke 
%MLP ba algoritm BP ast,dar shabke asabi az 2 vorudi
%10 neuron dar laye makhfi va 1 neuron dar laye khoruji estefade shode ast.
%in barname az tabe,e sigmoid estefade mikonad ke dar dakhele matlab vojud
%nadarad,manzur az mydata2 baaze nemune haye [-2,2] ast,manzur az mydata100nemune
%tedad nemune haye amuzeshi va azmayeshi ast ke 100 adad hastand.
clear all;
close all;
load mydata % Data1 Data2 Datatest mojud hastand va baraye har kodaam be 
            % tartib 350 ,350 va 300 nemune mojud ast,baraye estefade az
            %baaze haye digar mydata10 va... ra az inja taghiir dahid.
            
wsinput=rand(3,10); wsoutput=rand(10,1); %laye makhfi shamel 10 neuron,
                                         %laye khoruji shamel 1 neuron
a=1; % Alfa baraye tabe'e fa'al saaz
landa=0.9; % Nerkhe yadgiri
rate=0.9; % Zarib update nerkhe yadgiri

for count=1:200 % tedad 200 epoch amuzesh kole dade ha
  error=0;
for s=1: size(Data1,1) %amuzesh be ezaye hame Data1
  d=0; %khoruji matlub
   inputlayer=Data1(s,:)*wsinput;  %emale vazn random be Data1
   fHidden= sigmoid(inputlayer,a);  %khoruji laye makhfi ba emal be sigmoid
   outputlayer=fHidden*wsoutput;  %mohasebe khoruji laye aval
   fOutput=sigmoid(outputlayer, a);  %mohasebe khoruji taabe 
%**************************************************

for i=1:size(fOutput,2) 
  outputdelta(i)=fOutput(i)*(1-fOutput(i))*(d-fOutput(i)); % khojuri matlub 0 ast.
end % (d-fOutput(i)) signal yadgiri ast
for i=1:size(fHidden,2) 
  hiddendeltaw(i,1)=landa*outputdelta(1)*fHidden(i);
end
for j=1:size(fHidden,2)
  hiddenlayerdelta(j)=a*fHidden(j)*(1-fHidden(j))*(outputdelta(1)*wsoutput(j,1));
end
for i=1:size(hiddendeltaw,1)   %eslahe vazn
  wsoutput(i)=wsoutput(i)+ hiddendeltaw(i,1);
end
inputdeltaw=landa*(hiddenlayerdelta'*Data1(s,:));
wsinput=wsinput+inputdeltaw';
error=error+power(d-fOutput(1),2); 
end;

%=====================================================

for s=1: size(Data2,1) %amuzesh be ezaye hame Data2,marahel daghighan hamanand marahel Data1 ast
  d=0;
   inputlayer=Data2(s,:)*wsinput;
   fHidden= sigmoid(inputlayer,a);
   outputlayer=fHidden*wsoutput;
   fOutput=sigmoid(outputlayer, a);
%**************************************************
for i=1:size(fOutput,2)
  outputdelta(i)=a*fOutput(i)*(1-fOutput(i))*(d-fOutput(i)); 
end % 
for i=1:size(fHidden,2)
  hiddendeltaw(i,1)=landa*outputdelta(1)*fHidden(i);
end
for j=1:size(fHidden,2)
  hiddenlayerdelta(j)=a*fHidden(j)*(1-fHidden(j))*(outputdelta(1)*wsoutput(j,1));
end
for i=1:size(hiddendeltaw,1)
  wsoutput(i)=wsoutput(i)+ hiddendeltaw(i,1);
end
  inputdeltaw=landa*(hiddenlayerdelta'*Data2(s,:));
  wsinput=wsinput+inputdeltaw';
  error=error+power(d-fOutput(1),2);
end;
if landa>0.1 % sharte tavaghofe landa
 landa=landa*rate;
end;

trainerror(count)=error; %mohasebe monhani khata baraye shabake asabi
end;

plot(trainerror); %tasvire monhani khata
title('Validation Error for multiplication');
xlabel('Training epoch');
ylabel('Error');
%azmayeshe shabake asabi ba Datatest
inputlayer=Datatest(:,:)*wsinput; fHidden= sigmoid(inputlayer,a);
outputlayer=fHidden*wsoutput; fOutput=sigmoid(outputlayer, a)
