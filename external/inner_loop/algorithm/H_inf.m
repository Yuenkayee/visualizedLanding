classdef H_inf < matlab.System
    % H_inf
    % MATLAB System implementation of the 4-input/4-output H-infinity
    % controller with manual D-K iteration for the helicopter model
    %
    %   x = [u, w, q, theta, v, p, r, phi, psi]'
    %   delta = [cyc_lat, cyc_lon, coll, ped]'
    %
    % Controlled outputs are selected by ControlledStateIndices.
    % For example, ControlledStateIndices = [1 2 5 7] gives
    %   y = [u, w, v, r]'
    %
    % Set the MATLAB System block simulation mode to Interpreted execution.
    % The controller is synthesized once in setupImpl, discretized, and
    % then propagated online as a discrete dynamic controller.
    %
    % Inputs
    %   r : 4x1 reference vector, ordered by ControlledStateIndices
    %   y : 4x1 measured vector, ordered by ControlledStateIndices
    %
    % Output
    %   delta : 4x1 control vector
    %           [cyc_lat; cyc_lon; coll; ped]
    %
    % By default, every control-input perturbation is saturated to the
    % interval [-0.01, 0.01] before it is sent to the plant.

    properties (Nontunable)
        % User-supplied continuous-time plant matrices.
        % Expected dimensions:
        %   PlantA : 9-by-9
        %   PlantB : 9-by-4
        %
        % ControlledStateIndices defines the four states to be controlled,
        % in the same order as the 4x1 command and feedback vectors.
        % Example: [1 2 5 7] gives y = [u; w; v; r].
        % The output matrices are generated internally as
        %   PlantC(i,ControlledStateIndices(i)) = 1
        %   PlantD = zeros(4,4)

        
        PlantA = zeros(9,9)
        PlantB = zeros(9,4)
        ControlledStateIndices = [1 2 5 7]

        SampleTime = 0.02
        DKIterations = 10
        SelectedIteration = 6
        FrequencyGrid = logspace(-3,3,300)
        DiscretizationMethod = 'tustin'

        % Performance-weight parameters.
        PerformanceA = 0.005
        PerformanceM = 2
        PerformanceBandwidth = 1

        % Input multiplicative-uncertainty weight
        % Wi(s) = (s + UncertaintyZero) /
        %         (UncertaintyDenS*s + UncertaintyDen0)
        UncertaintyZero = 0.2
        UncertaintyDenS = 0.5
        UncertaintyDen0 = 1

        % Limit all four control-input perturbations to +/-0.01.
        % Output order: [cyc_lat; cyc_lon; coll; ped].
        EnableInputSaturation = true
        InputLowerLimit = -0.01*ones(4,1)
        InputUpperLimit =  0.01*ones(4,1)
    end

    properties (Access = private)
        Ad
        Bd
        Cd
        Dd
        xK
        ControllerOrder
    end

    methods (Access = protected)
        function setupImpl(obj)
            validateattributes(obj.SampleTime,{'numeric'}, ...
                {'scalar','real','finite','positive'});
            validateattributes(obj.DKIterations,{'numeric'}, ...
                {'scalar','integer','>=',2});
            validateattributes(obj.SelectedIteration,{'numeric'}, ...
                {'scalar','integer','>=',2,'<=',obj.DKIterations});

            obj.validatePlantMatrices();

            Kfinal = obj.synthesizeController();
            Kd = c2d(ss(Kfinal),obj.SampleTime, ...
                obj.DiscretizationMethod);
            [obj.Ad,obj.Bd,obj.Cd,obj.Dd] = ssdata(Kd);

            obj.ControllerOrder = size(obj.Ad,1);
            obj.xK = zeros(obj.ControllerOrder,1);
        end

        function delta = stepImpl(obj,r,y)
            r = double(r(:));
            y = double(y(:));

            if numel(r) ~= 4 || numel(y) ~= 4
                error('H_inf:SignalDimension', ...
                    ['Both r and y must contain exactly four elements, ' ...
                     'ordered by ControlledStateIndices.']);
            end

            e = r - y;

            delta = obj.Cd*obj.xK + obj.Dd*e;
            obj.xK = obj.Ad*obj.xK + obj.Bd*e;

            if obj.EnableInputSaturation
                lower = obj.InputLowerLimit(:);
                upper = obj.InputUpperLimit(:);

                if numel(lower) ~= 4 || numel(upper) ~= 4
                    error('H_inf:LimitDimension', ...
                        ['InputLowerLimit and InputUpperLimit must each ' ...
                         'contain four elements.']);
                end

                delta = min(max(delta,lower),upper);
            end
        end

        function resetImpl(obj)
            if isempty(obj.ControllerOrder)
                obj.xK = zeros(0,1);
            else
                obj.xK = zeros(obj.ControllerOrder,1);
            end
        end

        function num = getNumInputsImpl(~)
            num = 2;
        end

        function num = getNumOutputsImpl(~)
            num = 1;
        end

        function sizeOut = getOutputSizeImpl(~)
            sizeOut = [4,1];
        end

        function typeOut = getOutputDataTypeImpl(~)
            typeOut = 'double';
        end

        function complexOut = isOutputComplexImpl(~)
            complexOut = false;
        end

        function fixedOut = isOutputFixedSizeImpl(~)
            fixedOut = true;
        end

        function sts = getSampleTimeImpl(obj)
            sts = createSampleTime(obj, ...
                'Type','Discrete', ...
                'SampleTime',obj.SampleTime);
        end

        function flag = supportsMultipleInstanceImpl(~)
            flag = true;
        end
    end

    methods (Access = private)
        function validatePlantMatrices(obj)
            Aplant = obj.PlantA;
            Bplant = obj.PlantB;
            idx = obj.ControlledStateIndices(:)';

            if ~isequal(size(Aplant),[9,9])
                error('H_inf:PlantA', ...
                    'PlantA must have size 9-by-9.');
            end

            if ~isequal(size(Bplant),[9,4])
                error('H_inf:PlantB', ...
                    'PlantB must have size 9-by-4.');
            end

            if numel(idx) ~= 4
                error('H_inf:ControlledStateIndices', ...
                    ['ControlledStateIndices must contain exactly four ' ...
                     'state indices.']);
            end

            if any(idx < 1) || any(idx > 9) || any(idx ~= round(idx))
                error('H_inf:ControlledStateIndices', ...
                    ['ControlledStateIndices must contain integer ' ...
                     'indices between 1 and 9.']);
            end

            if numel(unique(idx)) ~= 4
                error('H_inf:ControlledStateIndices', ...
                    'ControlledStateIndices must not contain repeated indices.');
            end
        end

        function [PlantC,PlantD] = buildOutputMatrices(obj)
            idx = obj.ControlledStateIndices(:)';
            PlantC = zeros(4,9);
            for i = 1:4
                PlantC(i,idx(i)) = 1;
            end
            PlantD = zeros(4,4);
        end

        function Kfinal = synthesizeController(obj)
            % Four-input/four-output plant:
            % input  = [cyc_lat, cyc_lon, coll, ped]'
            % output = selected states defined by ControlledStateIndices
            [PlantC,PlantD] = obj.buildOutputMatrices();
            G = ss(obj.PlantA,obj.PlantB,PlantC,PlantD);

            % Performance weighting function.
            wp = tf( ...
                [1/obj.PerformanceM,obj.PerformanceBandwidth], ...
                [1,obj.PerformanceBandwidth*obj.PerformanceA]);
            Wp = wp*eye(4);

            % Multiplicative input-uncertainty weighting function.
            wi = tf( ...
                [1,obj.UncertaintyZero], ...
                [obj.UncertaintyDenS,obj.UncertaintyDen0]);
            Wi = wi*eye(4);

            % Generalized plant inputs:
            %   [ydel(4); w(4); uctrl(4)]
            % Generalized plant outputs:
            %   [Wi*uctrl;
            %    Wp*(w + G*(uctrl + ydel));
            %    -(G*(uctrl + ydel) + w)]
            zero44 = tf(zeros(4));
            eye44 = tf(eye(4));

            P11 = zero44;
            P12 = zero44;
            P13 = Wi;

            P21 = Wp*G;
            P22 = Wp;
            P23 = Wp*G;

            P31 = -G;
            P32 = -eye44;
            P33 = -G;

            P = minreal([P11,P12,P13; ...
                         P21,P22,P23; ...
                         P31,P32,P33]);

            nMeas = 4;
            nCtrl = 4;
            w = obj.FrequencyGrid;

            % Four scalar input-uncertainty blocks and one full 4-by-4
            % fictitious performance block.
            blk = [1 1; 1 1; 1 1; 1 1; 4 4];

            % Initial K step.
            Dscale = append(1,1,1,1,tf(eye(4)),tf(eye(4)));
            [K,~,~,~] = hinfsyn(Dscale*P/Dscale,nMeas,nCtrl);

            % Initial D step.
            Nf = frd(lft(P,K),w);
            [~,muInfo] = mussv(Nf,blk);

            Kfinal = K;

            for iteration = 2:obj.DKIterations
                [dsysl,~] = mussvunwrap(muInfo);

                % Normalize the D scaling by the fourth scalar
                % uncertainty channel.
                dsysl = dsysl/dsysl(4,4);
                di = fitfrd(genphase(dsysl(1,1)),3);

                Di = append(di,di,di,di, ...
                    tf(eye(4)),tf(eye(4)));

                [Ki,~,~,~] = hinfsyn(Di*P/Di,nMeas,nCtrl);

                Nf = frd(lft(P,Ki),w);
                [~,muInfo] = mussv(Nf,blk);

                if iteration == obj.SelectedIteration
                    Kfinal = Ki;
                end
            end
        end
    end
end