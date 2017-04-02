function val = isGraphDegG1(G)
%ISGRAPHDEGG1 - checks to see if a graph's in and out degrees are greater
%than 1 for all nodes in the graph.  If so, returns 1, otherwise returns 0
% Inputs
%  G - the adjacency matrix
% 
% Outputs
%  val - 0 or 1 depending on whether the in-degrees or out-degrees > 1
%
%**************************************************************************
%* 
%* Copyright (C) 2017  Kiran Karra <kiran.karra@gmail.com>
%*
%* This program is free software: you can redistribute it and/or modify
%* it under the terms of the GNU General Public License as published by
%* the Free Software Foundation, either version 3 of the License, or
%* (at your option) any later version.
%*
%* This program is distributed in the hope that it will be useful,
%* but WITHOUT ANY WARRANTY; without even the implied warranty of
%* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%* GNU General Public License for more details.
%*
%* You should have received a copy of the GNU General Public License
%* along with this program.  If not, see <http://www.gnu.org/licenses/>.
%* 
%**************************************************************************

indeg = sum(G,1);
outdeg = sum(G,2);
if(any(indeg>1) || any(outdeg>1))
    val = 1;
else
    val = 0;
end
    
end