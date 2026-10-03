

classdef multi_trim < matlab.System
    % multi_trim
    % Trim-point interpolation module for the multi-trim-point LPV
    % helicopter model.
    %
    % The scheduling variable is the NED down-direction velocity v_down,
    % in knots.
    %
    % Trim data:
    %   v_down =  0 kt -> trim_0  = [V_0;  euler_0]
    %   v_down =  5 kt -> trim_5  = [V_5;  euler_5]
    %   v_down = 10 kt -> trim_10 = [V_10; euler_10]
    %
    % The trim target is computed by piecewise-linear interpolation:
    %
    %   if 0 <= v_down <= 5:
    %       trim(v_down) = (1-lambda)*trim_0 + lambda*trim_5
    %       lambda = v_down/5
    %
    %   if 5 < v_down <= 10:
    %       trim(v_down) = (1-lambda)*trim_5 + lambda*trim_10
    %       lambda = (v_down-5)/5
    %
    % Here:
    %   V_i     : 3x1 velocity trim vector
    %   euler_i : 3x1 Euler-angle trim vector
    %
    % MATLAB System block input:
    %   input 1: v_down, scalar scheduling variable in knots
    %
    % MATLAB System block output:
    %   output 1: trim_target = [V_trim; euler_trim], 6x1

    properties (Nontunable)
        % Trim vectors at each scheduling point.
        % Each V_* and euler_* must contain 3 elements.


        V_10 = zeros(3,1)
        euler_10 = zeros(3,1)

        V_5 = zeros(3,1)
        euler_5 = zeros(3,1)

        V_0 = zeros(3,1)
        euler_0 = zeros(3,1)

        % If true, v_down is clamped to the interpolation range [0,10].
        % If false, linear extrapolation is used outside the range.
        ClampSchedulingVariable = true
    end

    methods (Access = protected)
        function setupImpl(obj)
            obj.validateParameters();
        end

        function trimPos = stepImpl(obj,v_down)
            v_down = double(v_down);

            if ~isscalar(v_down) || ~isfinite(v_down)
                error('multi_trim:SchedulingVariable', ...
                    'v_down must be a finite scalar in knots.');
            end

            trimPos = obj.interpolateTrim(v_down);
        end

        function num = getNumInputsImpl(~)
            num = 1;
        end

        function num = getNumOutputsImpl(~)
            num = 1;
        end

        function sizeOut = getOutputSizeImpl(~)
            sizeOut = [6,1];
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

        function flag = supportsMultipleInstanceImpl(~)
            flag = true;
        end
    end

    methods (Access = private)
        function validateParameters(obj)
            obj.validateVector3(obj.V_10,'V_10');
            obj.validateVector3(obj.euler_10,'euler_10');
            obj.validateVector3(obj.V_5,'V_5');
            obj.validateVector3(obj.euler_5,'euler_5');
            obj.validateVector3(obj.V_0,'V_0');
            obj.validateVector3(obj.euler_0,'euler_0');
        end

        function validateVector3(~,value,name)
            if numel(value) ~= 3
                error('multi_trim:TrimVectorDimension', ...
                    '%s must contain exactly 3 elements.',name);
            end
        end

        function trim_target = interpolateTrim(obj,v_down)
            trim_0 = [obj.V_0(:); obj.euler_0(:)];
            trim_5 = [obj.V_5(:); obj.euler_5(:)];
            trim_10 = [obj.V_10(:); obj.euler_10(:)];

            if obj.ClampSchedulingVariable
                v = min(max(v_down,0),10);
            else
                v = v_down;
            end

            knot_mid = obj.V_5(3) / 0.514444;
            knot_end = obj.V_10(3) / 0.514444;
            if v <= knot_mid
                lambda = v / knot_mid;
                trim_target = (1 - lambda) * trim_0 + lambda * trim_5;
            else
                lambda = (v - knot_mid) / (knot_end - knot_mid);
                trim_target = (1 - lambda) * trim_5 + lambda * trim_10;
            end
        end
    end
end