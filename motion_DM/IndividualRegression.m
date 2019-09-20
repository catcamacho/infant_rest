function [RegressionStruct] = IndividualRegression(Data,varargin)

Parser = inputParser;
addRequired(Parser,'Data',@isnumeric)
addOptional(Parser,'Shared',{},@iscell)
addOptional(Parser,'Individual',{},@iscell)
addOptional(Parser,'PartialFitVariables',{},@iscell)
addOptional(Parser,'CensorIndex',true([size(Data,1),1]),@islogical)
parse(Parser,Data,varargin{:});

ObservationCount = size(Data,1);
VariableCount = size(Data,2);

CensorIndex = Parser.Results.CensorIndex;

SharedLabelCell = Parser.Results.Shared(1:2:end);
SharedDataCell = Parser.Results.Shared(2:2:end);
SharedColumnCounts = cellfun(@(x) size(x,2),SharedDataCell);

IndividualLabelCell = Parser.Results.Individual(1:2:end);
IndividualDataCell = Parser.Results.Individual(2:2:end);
IndividualColumnCounts = numel(IndividualLabelCell);

PartialFitLabels = Parser.Results.PartialFitVariables;
ComputePartialFit = ~isempty(PartialFitLabels);

FullColumnCount = sum(vertcat(SharedColumnCounts(:),IndividualColumnCounts));

WeightMatrix = zeros(FullColumnCount,VariableCount);
FitMatrix = zeros(size(Data));

PartialFitColumnSelectionIDX = [];
if ComputePartialFit
    PartialFitMatrix = zeros(size(Data));
    
    PartialFitColumnSelectionIDX = false([1,FullColumnCount]);
    EntryStartIDX = 0;
    EntryEndIDX = 0;
    for SharedIDX = 1:numel(SharedLabelCell)
        EntryStartIDX = EntryEndIDX + 1;
        EntryEndIDX = EntryStartIDX + SharedColumnCounts(SharedIDX) - 1;
        if ismember(SharedLabelCell{SharedIDX},PartialFitLabels)
            PartialFitColumnSelectionIDX(EntryStartIDX:EntryEndIDX) = true;
        end
    end
    
    for IndividualIDX = 1:numel(IndividualLabelCell)
        EntryStartIDX = EntryEndIDX + 1;
        EntryEndIDX = EntryStartIDX;
        if ismember(IndividualLabelCell{IndividualIDX},PartialFitLabels)
            PartialFitColumnSelectionIDX(EntryStartIDX:EntryEndIDX) = true;
        end
    end
    
else
    PartialFitMatrix = [];
end


SharedXMat = horzcat(SharedDataCell{:});

if IndividualColumnCounts == 0
    
    WeightMatrix = SharedXMat(CensorIndex,:) \ Data(CensorIndex,:);
    FitMatrix = SharedXMat * WeightMatrix;
    
    if ComputePartialFit
        PartialFitMatrix = ThisXMat(:,PartialFitColumnSelectionIDX) *  ThisWeights(PartialFitColumnSelectionIDX,:);
    end
    
else
    
    parfor VarIDX = 1:VariableCount
        
        % Pull together all of the individual regressors for this variable
        ThisIndividualXMat = zeros(ObservationCount,IndividualColumnCounts);
        for IndividualIDX = 1:IndividualColumnCounts
            ThisIndividualXMat(:,IndividualIDX) = IndividualDataCell{IndividualIDX}(:,VarIDX);
        end
        
        ThisXMat = horzcat(SharedXMat,ThisIndividualXMat);
        ThisWeights = ThisXMat(CensorIndex,:) \ Data(CensorIndex,VarIDX);
        WeightMatrix(:,VarIDX) = ThisWeights;
        FitMatrix(:,VarIDX) = ThisXMat * WeightMatrix(:,VarIDX);
        
        if ComputePartialFit
            PartialFitMatrix(:,VarIDX) = ThisXMat(:,PartialFitColumnSelectionIDX) *  ThisWeights(PartialFitColumnSelectionIDX);
        end
        
        
    end
    
    
end



ColStartIDX = 0;
ColEndIDX = 0;

for LabelIDX = 1:numel(SharedLabelCell)
    
    ThisLabel = SharedLabelCell{LabelIDX};
    ThisColCount = size(SharedDataCell{LabelIDX},2);
    ColStartIDX = ColEndIDX + 1;
    ColEndIDX = ColStartIDX + ThisColCount - 1;
    WeightStruct.(ThisLabel) = WeightMatrix(ColStartIDX:ColEndIDX,:);
    
end

for LabelIDX = 1:numel(IndividualLabelCell)
    ThisLabel = IndividualLabelCell{LabelIDX};
    ThisColCount = 1;
    ColStartIDX = ColEndIDX + 1;
    ColEndIDX = ColStartIDX + ThisColCount - 1;
    WeightStruct.(ThisLabel) = WeightMatrix(ColStartIDX:ColEndIDX,:);
end


ResidualsMatrix = Data - FitMatrix;
RegressionStruct.Weights = WeightStruct;
RegressionStruct.Fits = FitMatrix;
RegressionStruct.Resid = ResidualsMatrix;
RegressionStruct.PartialFit = PartialFitMatrix;

end