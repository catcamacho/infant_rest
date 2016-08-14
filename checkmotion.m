function checkmotion

%%
% Things to add:
% mean displacement plot
% Maximum displacement plot
% Re-running spm_align with new data set
%%
fp = '/Users/myelin/Box Sync/SPM_practice/002rest/rest';
threshhold = 0.2;
minTimeAnalyze = 300;
TR = 2.5;

usabilityThresh = minTimeAnalyze/TR;

files = spm_select('ExtList', fp, 'vol*');
cd(fp);
spm_realign(files);

motion = textscan(fopen([fp '/rp_vol0002.txt'],'r'), '%16f%16f%16f%16f%16f%f%[^\n\r]', 'Delimiter', '', 'WhiteSpace', '',  'ReturnOnError', false);

d = dir([fp '/vol*']);
fileslist = transpose({d.name});

x = abs(motion{1});
y = abs(motion{2});
z = abs(motion{3});

time = 1:size(x);

relx(1,1) = 0;
rely(1,1) = 0;
relz(1,1) = 0;
for i = 2:size(x);
    relx(i,:) = abs(x(i,:) - x(i-1,:));
end
for i = 2:size(y);
    rely(i,:) = abs(y(i,:) - y(i-1,:));
end
for i = 2:size(z);
    relz(i,:) = abs(z(i,:) - z(i-1,:));
end

roll = motion{4};
pitch = motion{5};
yaw = motion {6};

figure
absolute_displacement = plot(time,x,time,y,time,z);
title('absolute Displacement')
xlabel('volume number')
ylabel('mm displaced')
grid on
grid minor

figure
relative_displacement = plot(time,relx,time,rely,time,relz);
title('relative Displacement')
xlabel('volume number')
ylabel('mm displaced')
grid on
grid minor

excessX = relx>=threshhold;
excessY = rely>=threshhold;
excessZ = relz>=threshhold;
excess = excessX + excessY + excessZ;

volsToCensor = fileslist(excess>0)

if (size(time) - size(volsToCensor)) >= usabilityThresh
    verdict = sprintf(['Subject is not usable. Too many volumes with motion greater than ' num2str(threshhold) 'mm of motion.'])
else
    verdict = sprintf(['Subject is usable! ' num2str(length(volsToCensor)) ' volumes with motion greater than ' num2str(threshhold) 'mm of motion.'])
end

end