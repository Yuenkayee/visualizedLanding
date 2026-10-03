classdef ship_motion_model_2 < matlab.System

    %SHIP_MOTION_MODEL_2 Six-DOF ship model based on equation (5).
    %
    % State convention (body axes x-forward, y-right, z-down):
    %   eta = [x; y; z; phi; theta; psi]       inertial position/attitude
    %   nu  = [u; v; w; p; q; r]               body-fixed velocity
    %
    % Inputs:
    %   eta, nu, tau, F_DI                    four 6-by-1 vectors
    % Outputs:
    %   eta_dot, nu_dot                       two 6-by-1 vectors
    %
    % The implemented equations are
    %   (M + A)*nu_dot + C(nu)*nu + B(nu)*nu + g(eta) = tau + F_DI
    %   eta_dot = J(eta)*nu
    %
    % C(nu) is formed from the constant total inertia M+A.  Damping and
    % restoring terms are represented as
    %   B(nu)*nu = LinearDamping*nu ...
    %              + QuadraticDamping*(abs(nu).*nu)
    %   g(eta)   = RestoringMatrix*(eta - RestoringEquilibrium)
    % This block returns derivatives; use two Integrator blocks (for nu
    % and eta) to perform the time-domain integration described after (5).
    % Default matrices are preliminary estimates for the Fassmer OPV 70
    % (70.2 m, 11.0 m beam, 3.5 m draught, about 1200 t displacement).

    properties (Nontunable)
        % Rigid-body mass/inertia matrix M_S [kg, kg*m, kg*m^2].

        MassMatrix = diag([1.2e6, 1.2e6, 1.2e6, ...
            1.987788e7, 3.69603e8, 3.69603e8])

        % Hydrodynamic added-mass matrix A_S.

        AddedMass = diag([6.0e4, 1.2e6, 4.08e6, ...
            8.945046e6, 1.108809e9, 1.2936105e8])

        % Linear part of the hydrodynamic damping matrix B_S(nu).

        LinearDamping = diag([2.1e4, 9.6e4, 2.772073661e6, ...
            3.608989399e6, 5.596093381e8, 1.6632135e7])

        % Quadratic damping coefficient matrix.  Set to zero when the
        % strip-theory damping matrix is entirely linear.

        QuadraticDamping = zeros(6)

        % Linear hydrostatic restoring matrix.

        RestoringMatrix = diag([0, 0, 5.821509381e6, ...
            1.765197e7, 2.353596e9, 0])

        % Position/attitude at which the restoring load is zero.

        RestoringEquilibrium = zeros(6, 1)

        % Minimum allowed |cos(theta)| in the Euler-rate transformation.

        EulerCosineTolerance = 1.0e-8
    end

    methods
        function obj = ship_motion_model_2(varargin)
            % Support either a parameter structure or name-value pairs.
            if nargin == 1 && isstruct(varargin{1})
                parameters = varargin{1};
                names = fieldnames(parameters);
                for k = 1:numel(names)
                    name = names{k};
                    if ~isprop(obj, name)
                        error('ship_motion_model_2:UnknownParameter', ...
                            'Unknown parameter field ''%s''.', name);
                    end
                    obj.(name) = parameters.(name);
                end
            elseif nargin > 0
                setProperties(obj, nargin, varargin{:});
            end
        end

        function [etaDot, nuDot] = derivatives(obj, eta, nu, tau, F_DI)
            %DERIVATIVES Evaluate equation (5) outside a System block.
            obj.validateParameters();
            obj.validateVector(eta, 'eta');
            obj.validateVector(nu, 'nu');
            obj.validateVector(tau, 'tau');
            obj.validateVector(F_DI, 'F_DI');
            [etaDot, nuDot] = obj.evaluateEquation( ...
                double(eta(:)), double(nu(:)), double(tau(:)), double(F_DI(:)));
        end

        function C = coriolisMatrix(obj, nu)
            %CORIOLISMATRIX Return C_S(nu) for the configured total inertia.
            obj.validateParameters();
            obj.validateVector(nu, 'nu');
            totalMass = double(obj.MassMatrix) + double(obj.AddedMass);
            C = obj.makeCoriolis(totalMass, double(nu(:)));
        end

        function J = kinematicMatrix(obj, eta)
            %KINEMATICMATRIX Return the body-to-inertial matrix J(eta).
            obj.validateVector(eta, 'eta');
            J = obj.makeKinematicMatrix(double(eta(:)));
        end
    end

    methods (Static)
        function parameters = nominal70mParameters()
            %NOMINAL70MPARAMETERS Fassmer OPV 70 preliminary parameter set.
            %
            % Intended use:
            %   p = ship_motion_model_2.nominal70mParameters();
            %   model = ship_motion_model_2(p);
            %
            % Assumptions:
            %   seawater density             1025 kg/m^3
            %   length / beam / draught      70.2 / 11.0 / 3.5 m
            %   displacement mass            1.2e6 kg
            %   waterplane coefficient       0.75 (engineering estimate)
            %   transverse metacentric height 1.5 m (engineering estimate)
            %   longitudinal metacentric height 200 m (estimated from Awp)
            %   CG is the body-frame origin; port-starboard symmetry
            %
            % The inertia radii follow common preliminary-test estimates:
            % k_xx = 0.37*B and k_yy = k_zz = 0.25*L.  Added mass is a
            % frequency-independent engineering approximation for an
            % initial time-domain/control model.  Replace AddedMass and
            % LinearDamping with strip-theory or identified data when
            % frequency-dependent hydrodynamic data are available.

            rho = 1025;
            gravity = 9.80665;
            length = 70.2;
            beam = 11.0;
            mass = 1.2e6;
            waterplaneCoefficient = 0.75;
            transverseGM = 1.5;
            longitudinalGM = 200;

            kRoll = 0.37 * beam;
            kPitch = 0.25 * length;
            kYaw = 0.25 * length;
            rigidInertia = mass * [kRoll^2, kPitch^2, kYaw^2];
            parameters.MassMatrix = diag([mass, mass, mass, rigidInertia]);

            % Diagonal preliminary added-mass estimate.  The entries are
            % positive physical added masses/inertias in this class's sign
            % convention (not the negative hydrodynamic derivatives).
            % The shallow, light OPV has relatively large heave/pitch
            % added inertia compared with its displacement mass. Ratios
            % are representative-frequency estimates, not RAO data. With
            % the restoring estimates below they give undamped periods of
            % about 5.98 s (heave), 8.03 s (roll), and 4.98 s (pitch).
            addedRatio = [0.05, 1.00, 3.40, 0.45, 3.00, 0.35];
            parameters.AddedMass = diag(addedRatio .* ...
                [mass, mass, mass, rigidInertia]);

            waterplaneArea = waterplaneCoefficient * length * beam;
            heaveStiffness = rho * gravity * waterplaneArea;
            rollStiffness = mass * gravity * transverseGM;
            pitchStiffness = mass * gravity * longitudinalGM;
            parameters.RestoringMatrix = diag( ...
                [0, 0, heaveStiffness, rollStiffness, pitchStiffness, 0]);
            parameters.RestoringEquilibrium = zeros(6, 1);

            % Equivalent linear damping.  Unrestored modes use decay time
            % constants; restored modes use 2*zeta*sqrt(M_eff*K).
            totalInertia = diag(parameters.MassMatrix + parameters.AddedMass);
            surgeTimeConstant = 60;
            swayTimeConstant = 25;
            yawTimeConstant = 30;
            zetaHeave = 0.25;
            zetaRoll = 0.08;
            zetaPitch = 0.15;
            damping = [ ...
                totalInertia(1) / surgeTimeConstant; ...
                totalInertia(2) / swayTimeConstant; ...
                2*zetaHeave*sqrt(totalInertia(3)*heaveStiffness); ...
                2*zetaRoll*sqrt(totalInertia(4)*rollStiffness); ...
                2*zetaPitch*sqrt(totalInertia(5)*pitchStiffness); ...
                totalInertia(6) / yawTimeConstant];
            parameters.LinearDamping = diag(damping);

            % The cited formulation uses linear strip-theory damping.  Keep
            % nonlinear damping disabled until decay-test/CFD data exist.
            parameters.QuadraticDamping = zeros(6);
            parameters.EulerCosineTolerance = 1.0e-8;
        end

        function parameters = fassmerOPV70Parameters()
            %FASSMEROPV70PARAMETERS Descriptive alias for the nominal set.
            parameters = ship_motion_model_2.nominal70mParameters();
        end

        function parameters = landingValidation70mParameters()
            %LANDINGVALIDATION70MPARAMETERS Lightly damped roll/pitch case.
            %
            % This profile retains the nominal vessel mass, added mass,
            % hydrostatic stiffness and natural periods, but reduces the
            % equivalent roll damping ratio from 0.08 to 0.025 and pitch
            % damping ratio from 0.15 to 0.05.  It is intended to generate
            % larger, longer-lasting deck angular motion for helicopter
            % landing-controller validation while preserving static
            % stability.
            %
            % Intended use:
            %   p = ship_motion_model_2.landingValidation70mParameters();
            %   model = ship_motion_model_2(p);

            parameters = ship_motion_model_2.nominal70mParameters();
            totalInertia = diag(parameters.MassMatrix + parameters.AddedMass);
            restoringDiagonal = diag(parameters.RestoringMatrix);

            zetaRoll = 0.025;
            zetaPitch = 0.05;
            parameters.LinearDamping(4, 4) = ...
                2*zetaRoll*sqrt(totalInertia(4)*restoringDiagonal(4));
            parameters.LinearDamping(5, 5) = ...
                2*zetaPitch*sqrt(totalInertia(5)*restoringDiagonal(5));
        end
    end

    methods (Access = protected)
        function setupImpl(obj, ~, ~, ~, ~)
            obj.validateParameters();
        end

        function [etaDot, nuDot] = stepImpl(obj, eta, nu, tau, F_DI)
            [etaDot, nuDot] = obj.evaluateEquation( ...
                double(eta(:)), double(nu(:)), double(tau(:)), double(F_DI(:)));
        end

        function validatePropertiesImpl(obj)
            obj.validateParameters();
        end

        function validateInputsImpl(obj, eta, nu, tau, F_DI)
            obj.validateVector(eta, 'eta');
            obj.validateVector(nu, 'nu');
            obj.validateVector(tau, 'tau');
            obj.validateVector(F_DI, 'F_DI');
        end

        function n = getNumInputsImpl(~)
            n = 4;
        end

        function n = getNumOutputsImpl(~)
            n = 2;
        end

        function [s1, s2] = getOutputSizeImpl(~)
            s1 = [6, 1];
            s2 = [6, 1];
        end

        function [d1, d2] = getOutputDataTypeImpl(~)
            d1 = 'double';
            d2 = 'double';
        end

        function [c1, c2] = isOutputComplexImpl(~)
            c1 = false;
            c2 = false;
        end

        function [f1, f2] = isOutputFixedSizeImpl(~)
            f1 = true;
            f2 = true;
        end

        function [n1, n2, n3, n4] = getInputNamesImpl(~)
            n1 = 'eta';
            n2 = 'nu';
            n3 = 'tau';
            n4 = 'F_DI';
        end

        function [n1, n2] = getOutputNamesImpl(~)
            n1 = 'eta_dot';
            n2 = 'nu_dot';
        end

        function icon = getIconImpl(~)
            icon = sprintf('Ship Motion\nEq. (5)');
        end

        function flag = supportsMultipleInstanceImpl(~)
            flag = true;
        end
    end

    methods (Access = private)
        function [etaDot, nuDot] = evaluateEquation(obj, eta, nu, tau, F_DI)
            totalMass = double(obj.MassMatrix) + double(obj.AddedMass);
            C = obj.makeCoriolis(totalMass, nu);
            dampingLoad = double(obj.LinearDamping) * nu + ...
                double(obj.QuadraticDamping) * (abs(nu) .* nu);
            restoringLoad = double(obj.RestoringMatrix) * ...
                (eta - double(obj.RestoringEquilibrium(:)));

            % Use a linear solve rather than explicitly inverting M_S+A_S.
            nuDot = totalMass \ ...
                (tau + F_DI - C * nu - dampingLoad - restoringLoad);
            etaDot = obj.makeKinematicMatrix(eta) * nu;
        end

        function C = makeCoriolis(obj, totalMass, nu)
            linearMomentum = totalMass(1:3, 1:3) * nu(1:3) + ...
                totalMass(1:3, 4:6) * nu(4:6);
            angularMomentum = totalMass(4:6, 1:3) * nu(1:3) + ...
                totalMass(4:6, 4:6) * nu(4:6);

            zero3 = zeros(3);
            momentumSkew = obj.skew(linearMomentum);
            C = [zero3,             -momentumSkew; ...
                 -momentumSkew, -obj.skew(angularMomentum)];
        end

        function J = makeKinematicMatrix(obj, eta)
            phi = eta(4);
            theta = eta(5);
            psi = eta(6);

            cPhi = cos(phi);   sPhi = sin(phi);
            cTheta = cos(theta); sTheta = sin(theta);
            cPsi = cos(psi);   sPsi = sin(psi);

            if abs(cTheta) < obj.EulerCosineTolerance
                error('ship_motion_model_2:EulerSingularity', ...
                    ['The pitch angle is too close to +/-pi/2 for the ', ...
                     'ZYX Euler-angle kinematics.']);
            end

            % Rotation from body-fixed coordinates to the inertial frame.
            R = [cPsi*cTheta, ...
                 -sPsi*cPhi + cPsi*sTheta*sPhi, ...
                  sPsi*sPhi + cPsi*cPhi*sTheta; ...
                 sPsi*cTheta, ...
                  cPsi*cPhi + sPhi*sTheta*sPsi, ...
                 -cPsi*sPhi + sTheta*sPsi*cPhi; ...
                 -sTheta, cTheta*sPhi, cTheta*cPhi];

            % Body angular velocity [p;q;r] to ZYX Euler-angle rates.
            T = [1, sPhi*tan(theta), cPhi*tan(theta); ...
                 0, cPhi,           -sPhi; ...
                 0, sPhi/cTheta,     cPhi/cTheta];
            J = [R, zeros(3); zeros(3), T];
        end

        function validateParameters(obj)
            obj.validateMatrix6(obj.MassMatrix, 'MassMatrix');
            obj.validateMatrix6(obj.AddedMass, 'AddedMass');
            obj.validateMatrix6(obj.LinearDamping, 'LinearDamping');
            obj.validateMatrix6(obj.QuadraticDamping, 'QuadraticDamping');
            obj.validateMatrix6(obj.RestoringMatrix, 'RestoringMatrix');
            obj.validateVector(obj.RestoringEquilibrium, 'RestoringEquilibrium');
            validateattributes(obj.EulerCosineTolerance, {'numeric'}, ...
                {'real', 'finite', 'scalar', 'positive', '<=', 1}, ...
                mfilename, 'EulerCosineTolerance');

            totalMass = double(obj.MassMatrix) + double(obj.AddedMass);
            symmetryScale = max(1, norm(totalMass, 'fro'));
            if norm(totalMass - totalMass.', 'fro') > 1.0e-10 * symmetryScale
                error('ship_motion_model_2:NonSymmetricMass', ...
                    'MassMatrix + AddedMass must be symmetric.');
            end
            [~, factorizationFlag] = chol(totalMass);
            if factorizationFlag ~= 0
                error('ship_motion_model_2:InvalidMass', ...
                    'MassMatrix + AddedMass must be positive definite.');
            end
        end

        function validateMatrix6(~, value, name)
            validateattributes(value, {'numeric'}, ...
                {'real', 'finite', 'size', [6, 6]}, mfilename, name);
        end

        function validateVector(~, value, name)
            validateattributes(value, {'numeric'}, ...
                {'real', 'finite', 'vector', 'numel', 6}, mfilename, name);
        end

        function S = skew(~, vector)
            S = [0,         -vector(3),  vector(2); ...
                 vector(3),  0,         -vector(1); ...
                -vector(2),  vector(1),  0];
        end
    end
end
