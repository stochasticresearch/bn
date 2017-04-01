classdef hcbnk1family
    %HCBNFAMILY - has all information related to a copula family in the
    %             context of the HCBN-K1 construction
    %
    %**********************************************************************
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
    %* along with this program.  If not, see <http://www.gnu.org/licenses/>
    %* 
    %**********************************************************************
    
    properties
        nodeName;
        nodeIdx;
        parentNodeNames;
        parentNodeIdxs;
        
        c_model_name;   % the name of the copula
        c_model_params; % the parameters associated with the copula
        C_discrete_integrate;   % the partial derivative of the copula function, w.r.t. the continuous nodes
        
    end
    
    methods
        function obj = hcbnk1family(nn, ni, pnn, pni, ...
                                    cc_model_name, cc_model_params, CC_discrete_integrate)
            obj.nodeName = nn;
            obj.nodeIdx = ni;
            obj.parentNodeNames = pnn;
            obj.parentNodeIdxs = pni;
            
            obj.c_model_name = cc_model_name;
            obj.c_model_params = cc_model_params;
            obj.C_discrete_integrate = CC_discrete_integrate;
        end
    end
    
end