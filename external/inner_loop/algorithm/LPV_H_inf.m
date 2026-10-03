

classdef LPV_H_inf < matlab.System
    % LPV_H_inf
    % Gain-scheduled LPV-H_infinity controller for the multi-trim-point
    % helicopter linear system.
    %
    % Scheduling variable:
    %   v_down, NED down-direction velocity, in knots.
    %
    % Trim-point models:
    %   v_down = 10 kt -> (A_10, B_10)
    %   v_down =  5 kt -> (A_5,  B_5)
    %   v_down =  0 kt -> (A_0,  B_0)
    %
    % For each trim point, this block synthesizes one 4-input/4-output
    % H-infinity / manual D-K controller using the same generalized plant
    % structure as H_inf.m. During simulation, all three local controllers
    % are propagated with the same tracking error e = r - y, and the final
    % control command is obtained by piecewise-linear interpolation of the
    % three local controller outputs according to v_down.
    %
    % State definition:
    %   x = [u, w, q, theta, v, p, r, phi, psi]'
    %
    % Input definition:
    %   delta = [cyc_lat, cyc_lon, coll, ped]'
    %
    % Controlled outputs are selected by ControlledStateIndices.
    % Example: ControlledStateIndices = [1 2 5 7] gives
    %   y = [u, w, v, r]'
    %
    % MATLAB System block inputs:
    %   input 1: r,      4x1 reference vector
    %   input 2: y,      4x1 measured output vector
    %   input 3: v_down, scalar scheduling variable in knots
    %
    % MATLAB System block output:
    %   output 1: delta, 4x1 scheduled control vector
    %
    % Set the MATLAB System block simulation mode to Interpreted execution.

    properties (Nontunable)
        % Trim-point plant matrices.


        A_10 = zeros(9,9)
        B_10 = zeros(9,4)
        A_5  = zeros(9,9)
        B_5  = zeros(9,4)
        A_0  = zeros(9,9)
        B_0  = zeros(9,4)

        % Four controlled state indices, ordered consistently with r and y.
        % Example: [1 2 5 7] gives y = [u; w; v; r].
        % Example: [1 4 8 9] gives y = [u; theta; phi; psi].


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

        % If true, v_down is clamped to the interpolation range [0,10].
        % If false, linear extrapolation is used outside the range.


        ClampSchedulingVariable = true

        % Output saturation for the scheduled control command.
        % Output order: [cyc_lat; cyc_lon; coll; ped].

        
        EnableInputSaturation = true
        InputLowerLimit = -0.01*ones(4,1)
        InputUpperLimit =  0.01*ones(4,1)
    end

    properties (Access = private)
        Ad0
        Bd0
        Cd0
        Dd0
        xK0

        Ad5
        Bd5
        Cd5
        Dd5
        xK5

        Ad10
        Bd10
        Cd10
        Dd10
        xK10

        ControllerOrder0
        ControllerOrder5
        ControllerOrder10
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

            K0 = obj.synthesizeController(obj.A_0,obj.B_0);
            K5 = obj.synthesizeController(obj.A_5,obj.B_5);
            K10 = obj.synthesizeController(obj.A_10,obj.B_10);

            [obj.Ad0,obj.Bd0,obj.Cd0,obj.Dd0] = ...
                obj.discretizeController(K0);
            [obj.Ad5,obj.Bd5,obj.Cd5,obj.Dd5] = ...
                obj.discretizeController(K5);
            [obj.Ad10,obj.Bd10,obj.Cd10,obj.Dd10] = ...
                obj.discretizeController(K10);

            obj.ControllerOrder0 = size(obj.Ad0,1);
            obj.ControllerOrder5 = size(obj.Ad5,1);
            obj.ControllerOrder10 = size(obj.Ad10,1);

            obj.xK0 = zeros(obj.ControllerOrder0,1);
            obj.xK5 = zeros(obj.ControllerOrder5,1);
            obj.xK10 = zeros(obj.ControllerOrder10,1);
        end

        function delta = stepImpl(obj,r,y,v_down)
            r = double(r(:));
            y = double(y(:));
            v_down = double(v_down);

            if numel(r) ~= 4 || numel(y) ~= 4
                error('LPV_H_inf:SignalDimension', ...
                    ['Both r and y must contain exactly four elements, ' ...
                     'ordered by ControlledStateIndices.']);
            end

            if ~isscalar(v_down) || ~isfinite(v_down)
                error('LPV_H_inf:SchedulingVariable', ...
                    'v_down must be a finite scalar in knots.');
            end

            e = r - y;

            [delta0,obj.xK0] = obj.localControllerStep( ...
                obj.Ad0,obj.Bd0,obj.Cd0,obj.Dd0,obj.xK0,e);
            [delta5,obj.xK5] = obj.localControllerStep( ...
                obj.Ad5,obj.Bd5,obj.Cd5,obj.Dd5,obj.xK5,e);
            [delta10,obj.xK10] = obj.localControllerStep( ...
                obj.Ad10,obj.Bd10,obj.Cd10,obj.Dd10,obj.xK10,e);

            delta = obj.interpolateControl(v_down,delta0,delta5,delta10);

            if obj.EnableInputSaturation
                lower = obj.InputLowerLimit(:);
                upper = obj.InputUpperLimit(:);

                if numel(lower) ~= 4 || numel(upper) ~= 4
                    error('LPV_H_inf:LimitDimension', ...
                        ['InputLowerLimit and InputUpperLimit must each ' ...
                         'contain four elements.']);
                end

                delta = min(max(delta,lower),upper);
            end
        end

        function resetImpl(obj)
            obj.xK0 = zeros(obj.ControllerOrder0,1);
            obj.xK5 = zeros(obj.ControllerOrder5,1);
            obj.xK10 = zeros(obj.ControllerOrder10,1);
        end

        function num = getNumInputsImpl(~)
            num = 3;
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
            if ~isequal(size(obj.A_10),[9,9]) || ...
                    ~isequal(size(obj.A_5),[9,9]) || ...
                    ~isequal(size(obj.A_0),[9,9])
                error('LPV_H_inf:StateMatrixDimension', ...
                    'A_10, A_5 and A_0 must all have size 9-by-9.');
            end

            if ~isequal(size(obj.B_10),[9,4]) || ...
                    ~isequal(size(obj.B_5),[9,4]) || ...
                    ~isequal(size(obj.B_0),[9,4])
                error('LPV_H_inf:InputMatrixDimension', ...
                    'B_10, B_5 and B_0 must all have size 9-by-4.');
            end

            idx = obj.ControlledStateIndices(:)';

            if numel(idx) ~= 4
                error('LPV_H_inf:ControlledStateIndices', ...
                    ['ControlledStateIndices must contain exactly four ' ...
                     'state indices.']);
            end

            if any(idx < 1) || any(idx > 9) || any(idx ~= round(idx))
                error('LPV_H_inf:ControlledStateIndices', ...
                    ['ControlledStateIndices must contain integer ' ...
                     'indices between 1 and 9.']);
            end

            if numel(unique(idx)) ~= 4
                error('LPV_H_inf:ControlledStateIndices', ...
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

        function [Ad,Bd,Cd,Dd] = discretizeController(obj,K)
            Kd = c2d(ss(K),obj.SampleTime,obj.DiscretizationMethod);
            [Ad,Bd,Cd,Dd] = ssdata(Kd);
        end

        function [deltaNext,xKNext] = localControllerStep(~,Ad,Bd,Cd,Dd,xK,e)
            deltaNext = Cd*xK + Dd*e;
            xKNext = Ad*xK + Bd*e;
        end

        function delta = interpolateControl(obj,v_down,delta0,delta5,delta10)
            if obj.ClampSchedulingVariable
                v = min(max(v_down,0),10);
            else
                v = v_down;
            end

            if v <= 5
                lambda = v/5;
                delta = (1-lambda)*delta0 + lambda*delta5;
            else
                lambda = (v-5)/5;
                delta = (1-lambda)*delta5 + lambda*delta10;
            end
        end

        function Kfinal = synthesizeController(obj,Aplant,Bplant)
            % Four-input/four-output plant:
            % input  = [cyc_lat, cyc_lon, coll, ped]'
            % output = selected states defined by ControlledStateIndices
            [PlantC,PlantD] = obj.buildOutputMatrices();
            G = ss(Aplant,Bplant,PlantC,PlantD);

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