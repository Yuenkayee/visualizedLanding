classdef nedVelToBody < matlab.System
    % nedVelToBody
    % Convert NED-frame velocity [vn; ve; vd] to body-frame velocity [u; v; w].
    %
    % Coordinate convention:
    %   NED frame:  x_n North, y_n East, z_n Down
    %   Body frame: x_b forward, y_b right, z_b down
    %
    % Euler angle convention:
    %   phi   : roll angle,  rad
    %   theta : pitch angle, rad
    %   psi   : yaw angle,   rad
    %
    % Inputs:
    %   V_ned = [vn; ve; vd]
    %   Euler = [phi; theta; psi]
    %
    % Output:
    %   V_body = [u; v; w]

    methods (Access = protected)
        function V_body = stepImpl(~, V_ned, Euler)
            %#codegen
            % V_ned : [3 x 1] or [1 x 3]
            % Euler : [3 x 1] or [1 x 3], [phi; theta; psi]

            vn = V_ned(1);
            ve = V_ned(2);
            vd = V_ned(3);

            phi   = Euler(1);
            theta = Euler(2);
            psi   = Euler(3);

            cphi = cos(phi);
            sphi = sin(phi);

            ctheta = cos(theta);
            stheta = sin(theta);

            cpsi = cos(psi);
            spsi = sin(psi);

            % Direction cosine matrix from body frame to NED frame.
            C_nb = [ ...
                ctheta*cpsi,  sphi*stheta*cpsi - cphi*spsi,  cphi*stheta*cpsi + sphi*spsi;
                ctheta*spsi,  sphi*stheta*spsi + cphi*cpsi,  cphi*stheta*spsi - sphi*cpsi;
                -stheta,      sphi*ctheta,                    cphi*ctheta ];

            % NED frame to body frame.
            V_body = C_nb.' * [vn; ve; vd];
        end

        function resetImpl(~)
            % No discrete states to reset.
        end

        function num = getNumInputsImpl(~)
            num = 2;
        end

        function num = getNumOutputsImpl(~)
            num = 1;
        end

        function varargout = getInputNamesImpl(~)
            varargout = {'V_ned', 'Euler'};
        end

        function varargout = getOutputNamesImpl(~)
            varargout = {'V_body'};
        end

        function flag = isInputSizeMutableImpl(~, ~)
            flag = false;
        end

        function flag = isInputDataTypeMutableImpl(~, ~)
            flag = false;
        end

        function sz = getOutputSizeImpl(~)
            sz = [3 1];
        end

        function dt = getOutputDataTypeImpl(~)
            dt = 'double';
        end

        function cp = isOutputComplexImpl(~)
            cp = false;
        end

        function fs = isOutputFixedSizeImpl(~)
            fs = true;
        end
    end
end
