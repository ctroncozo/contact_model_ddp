import numpy as np
import crocoddyl
import example_robot_data
import pinocchio
from utils import plotSolution


class SimpleQuadrupedalGaitProblem:
    def __init__(self, rmodel, lfFoot, rfFoot, lhFoot, rhFoot):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Getting the frame id for all the legs
        self.lfrontFootId = self.rmodel.getFrameId(lfFoot)
        self.rfrontFootId = self.rmodel.getFrameId(rfFoot)
        self.lbackFootId = self.rmodel.getFrameId(lhFoot)
        self.rbackFootId = self.rmodel.getFrameId(rhFoot)
        # Defining default state
        q0 = self.rmodel.referenceConfigurations["standing"]
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.nsurf = np.array([0., 0., 1.])

    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)

        for i in supportFootIds:
            xref = crocoddyl.FrameTranslation(i, np.array([0., 0., 0.]))
            supportContactModel = crocoddyl.ContactModel3D(self.state, xref, self.actuation.nu, np.array([0., 50.]))
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        if isinstance(comTask, np.ndarray):
            comTrack = crocoddyl.CostModelCoMPosition(self.state, comTask, self.actuation.nu)
            costModel.addCost("comTrack", comTrack, 1e6)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
            frictionCone = crocoddyl.CostModelContactFrictionCone(
                self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                crocoddyl.FrameFrictionCone(i, cone), self.actuation.nu)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
        if swingFootTask is not None:
            for i in swingFootTask:
                xref = crocoddyl.FrameTranslation(i.id, i.placement.translation)
                footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
                costModel.addCost(self.rmodel.frames[i.id].name + "_footTrack", footTrack, 1e6)

        stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10.] * 6 + [1.] *
                                (self.rmodel.nv - 6))
        stateReg = crocoddyl.CostModelState(self.state, crocoddyl.ActivationModelWeightedQuad(stateWeights**2),
                                            self.rmodel.defaultState, self.actuation.nu)
        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        stateBounds = crocoddyl.CostModelState(
            self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub)),
            0 * self.rmodel.defaultState, self.actuation.nu)
        costModel.addCost("stateBounds", stateBounds, 1e3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model

    def createFootstepModels(self, comPos0, feetPos0, stepLength, stepHeight, timeStep, numKnots, supportFootIds,
                             swingFootIds):
        """ Action models for a footstep phase.

        :param comPos0, initial CoM position
        :param feetPos0: initial position of the swinging feet
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: time step
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :return footstep action models
        """
        numLegs = len(supportFootIds) + len(swingFootIds)
        comPercentage = float(len(swingFootIds)) / numLegs

        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i, p in zip(swingFootIds, feetPos0):
                # Defining a foot swing task given the step length
                # resKnot = numKnots % 2
                phKnots = numKnots / 2
                if k < phKnots:
                    dp = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight * k / phKnots])
                elif k == phKnots:
                    dp = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight])
                else:
                    dp = np.array(
                        [stepLength * (k + 1) / numKnots, 0., stepHeight * (1 - float(k - phKnots) / phKnots)])
                tref = p + dp

                swingFootTask += [crocoddyl.FramePlacement(i, pinocchio.SE3(np.eye(3), tref))]

            comTask = np.array([stepLength * (k + 1) / numKnots, 0., 0.]) * comPercentage + comPos0
            footSwingModel += [
                self.createSwingFootModel(timeStep, supportFootIds, comTask=comTask, swingFootTask=swingFootTask)
            ]

        # Action model for the foot switch
        footSwitchModel = self.createFootSwitchModel(supportFootIds, swingFootTask)

        # Updating the current foot position for next step
        comPos0 += [stepLength * comPercentage, 0., 0.]
        for p in feetPos0:
            p += [stepLength, 0., 0.]
        return footSwingModel + [footSwitchModel]

    def createFootSwitchModel(self, supportFootIds, swingFootTask, pseudoImpulse=False):
        """ Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the impulse model
        :return action model for a foot switch phase
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(supportFootIds, swingFootTask)
        else:
            return self.createImpulseModel(supportFootIds, swingFootTask)

    def createPseudoImpulseModel(self, supportFootIds, swingFootTask):
        """ Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact velocities.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return pseudo-impulse differential action model
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for i in supportFootIds:
            xref = crocoddyl.FrameTranslation(i, np.array([0., 0., 0.]))
            supportContactModel = crocoddyl.ContactModel3D(self.state, xref, self.actuation.nu, np.array([0., 50.]))
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
            frictionCone = crocoddyl.CostModelContactFrictionCone(
                self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                crocoddyl.FrameFrictionCone(i, cone), self.actuation.nu)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
        if swingFootTask is not None:
            for i in swingFootTask:
                xref = crocoddyl.FrameTranslation(i.frame, i.oMf.translation)
                vref = crocoddyl.FrameMotion(i.frame, pinocchio.Motion.Zero())
                footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
                impulseFootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
                costModel.addCost(self.rmodel.frames[i.frame].name + "_footTrack", footTrack, 1e7)
                costModel.addCost(self.rmodel.frames[i.frame].name + "_impulseVel", impulseFootVelCost, 1e6)
        stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10.] * self.rmodel.nv)
        stateReg = crocoddyl.CostModelState(self.state, crocoddyl.ActivationModelWeightedQuad(stateWeights ** 2),
                                            self.rmodel.defaultState, self.actuation.nu)
        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        # Use an Euler sympletic integrator to convert the differential action model into an action model.
        # Note that our solvers use action model.
        model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.)
        return model

    def createImpulseModel(self, supportFootIds, swingFootTask, JMinvJt_damping=1e-12, r_coeff=0.0):
        """ Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        # Creating a 3D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel3D(self.state, i)
            impulseModel.addImpulse(self.rmodel.frames[i].name + "_impulse", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                xref = crocoddyl.FrameTranslation(i.id, i.placement.translation)
                footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, 0)
                costModel.addCost(self.rmodel.frames[i.id].name + "_footTrack", footTrack, 1e7)
        stateWeights = np.array([1.] * 6 + [10.] * (self.rmodel.nv - 6) + [10.] * self.rmodel.nv)
        stateReg = crocoddyl.CostModelState(self.state, crocoddyl.ActivationModelWeightedQuad(stateWeights ** 2),
                                            self.rmodel.defaultState, 0)
        costModel.addCost("stateReg", stateReg, 1e1)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, costModel)
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model

    def createWalkingProblem(self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple walking gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rFrontFootPos0 = self.rdata.oMf[self.rfrontFootId].translation
        rBackFootPos0 = self.rdata.oMf[self.rbackFootId].translation

        lFrontFootPos0 = self.rdata.oMf[self.lfrontFootId].translation
        lBackFootPos0 = self.rdata.oMf[self.lbackFootId].translation
        comRef = (rFrontFootPos0 + rBackFootPos0 + lFrontFootPos0 + lBackFootPos0) / 4
        comRef[2] = np.asscalar(pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2])

        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [
            self.createSwingFootModel(
                timeStep,
                [self.lfrontFootId, self.rfrontFootId, self.lbackFootId, self.rbackFootId],
            ) for k in range(supportKnots)
        ]

        rbackStep = self.createFootstepModels(comRef, [rBackFootPos0], stepLength, stepHeight, timeStep, stepKnots,
                                               [self.lfrontFootId, self.rfrontFootId, self.lbackFootId], [self.rbackFootId])
        rfrontStep = self.createFootstepModels(comRef, [rFrontFootPos0], stepLength, stepHeight, timeStep, stepKnots,
                                               [self.lfrontFootId, self.lbackFootId, self.rbackFootId], [self.rfrontFootId])
        lbackStep = self.createFootstepModels(comRef, [lBackFootPos0], stepLength, stepHeight, timeStep, stepKnots,
                                           [self.lfrontFootId, self.rfrontFootId, self.rbackFootId], [self.lbackFootId])
        lfrontStep = self.createFootstepModels(comRef, [lFrontFootPos0], stepLength, stepHeight, timeStep, stepKnots,
                                           [self.rfrontFootId, self.lbackFootId, self.rbackFootId], [self.lfrontFootId])

        # Why do we need the double support? at leas for walking does not seem necessary, maybe for other gaits.
        #loco3dModel += doubleSupport + rbackStep + rfrontStep
        #loco3dModel += doubleSupport + lbackStep + lfrontStep
        loco3dModel += rbackStep + rfrontStep
        loco3dModel += lbackStep + lfrontStep
        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem


def run():
    # Loading the anymal model
    anymal = example_robot_data.load('anymal')

    # nq is the dimension fo the configuration vector representation
    # nv dimension of the velocity vector space

    # Defining the initial state of the robot
    q0 = anymal.model.referenceConfigurations['standing'].copy()
    v0 = pinocchio.utils.zero(anymal.model.nv)
    x0 = np.concatenate([q0, v0])

    # Setting up the 3d walking problem
    lfFoot, rfFoot, lhFoot, rhFoot = 'LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT'
    gait = SimpleQuadrupedalGaitProblem(anymal.model, lfFoot, rfFoot, lhFoot, rhFoot)

    cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
    walking = {
        'stepLength': 0.25,
        'stepHeight': 0.15,
        'timeStep': 1e-2,
        'stepKnots': 100,
        'supportKnots': 2
    }
    # Creating a walking problem
    ddp = crocoddyl.SolverFDDP(
        gait.createWalkingProblem(x0, walking['stepLength'], walking['stepHeight'], walking['timeStep'],
                                  walking['stepKnots'], walking['supportKnots']))
    plot = False
    display = False
    if display:
        # Added the callback functions
        display = crocoddyl.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
        ddp.setCallbacks(
            [crocoddyl.CallbackLogger(),
             crocoddyl.CallbackVerbose(),
             crocoddyl.CallbackDisplay(display)])


    # Solving the problem with the DDP solver
    xs = [anymal.model.defaultState] * (ddp.problem.T + 1)
    us = ddp.problem.quasiStatic([anymal.model.defaultState] * ddp.problem.T)
    ddp.solve(xs, us, 100, False, 0.1)

    if display:
        # Defining the final state as initial one for the next phase
        # Display the entire motion
        display = crocoddyl.GepettoDisplay(anymal, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
        display.displayFromSolver(ddp)
    # Plotting the entire motion
    if plot:
        plotSolution(ddp, figIndex=1, show=False)

        log = ddp.getCallbacks()[0]
        crocoddyl.plotConvergence(log.costs,
                                  log.u_regs,
                                  log.x_regs,
                                  log.grads,
                                  log.stops,
                                  log.steps,
                                  figTitle='walking',
                                  figIndex=3,
                                  show=True)
#https://memory-of-motion.github.io/summer-school/materials/memmo_summer_school_Crocoddyl_Carlos_Mastalli.pdf
run()