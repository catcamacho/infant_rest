function [Matrix] = GetLeadLagMatrix(X,Lead,Lag)

LeadLagAxis = [-abs(Lead):abs(Lag)];
[RowCount,ColCount] = size(X);
Matrix =  zeros([size(X,1),numel(LeadLagAxis)*ColCount]);

for ColIDX = 1:ColCount
    
    ColData = X(:,ColIDX);
    
    for LagIDX = 1:numel(LeadLagAxis)
        ThisLag = LeadLagAxis(LagIDX);
        MatrixColIDX = (ColIDX - 1) * numel(LeadLagAxis) + LagIDX;
        
        if ThisLag < 0
            DataSelectionStart = abs(ThisLag)+1;
            DataSelectionEnd = RowCount;  
            
            DataPlacementStart = 1;
            DataPlacementEnd = RowCount - abs(ThisLag);
        elseif ThisLag == 0
            DataSelectionStart = 1;
            DataSelectionEnd = RowCount; 
            
            DataPlacementStart = 1;
            DataPlacementEnd = RowCount;            
        else
           
            DataSelectionStart = 1;
            DataSelectionEnd = RowCount-ThisLag;              
            
            DataPlacementStart = ThisLag+1;
            DataPlacementEnd = RowCount;            
        end
            
        Matrix(DataPlacementStart:DataPlacementEnd,MatrixColIDX) = ColData(DataSelectionStart:DataSelectionEnd);

    end

end

end