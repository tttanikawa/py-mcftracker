import numpy as np
import scipy.linalg

class KalmanFilter(object):

    def __init__(self):
        ndim, dt = 2, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)

        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_position = 6.25
        self._std_velocity = 4.

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            world coordinates of a box (x,y)

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (4 dimensional) and covariance matrix (4x4
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            self._std_position,
            self._std_position,
            self._std_velocity,
            self._std_velocity
        ]

        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 4 dimensional mean vector of the object state at the previous time step.
        covariance : ndarray
            The 4x4 dimensional covariance matrix of the object state at the previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """

        std_pos = [
            self._std_position,
            self._std_position
        ]

        std_vel = [
            self._std_velocity,
            self._std_velocity
        ]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)

        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (4 dimensional array).
        covariance : ndarray
            The state's covariance matrix (4x4 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """

        std = [
            self._std_position,
            self._std_position
        ]

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
            
        return mean, covariance + innovation_cov
        
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (4 dimensional).
        covariance : ndarray
            The state's covariance matrix (4x4 dimensional).
        measurement : ndarray
            world coordinates of a box (x, y)

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """

        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T

        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=True):
            """Compute gating distance between state distribution and measurements.

            A suitable distance threshold can be obtained from `chi2inv95`. If
            `only_position` is False, the chi-square distribution has 4 degrees of
            freedom, otherwise 2.

            Parameters
            ----------
            mean : ndarray
                Mean vector over the state distribution (8 dimensional).
            covariance : ndarray
                Covariance of the state distribution (8x8 dimensional).
            measurements : ndarray
                An Nx4 dimensional matrix of N measurements, each in
                format (x, y, a, h) where (x, y) is the bounding box center
                position, a the aspect ratio, and h the height.
            only_position : Optional[bool]
                If True, distance computation is done with respect to the bounding
                box center position only.

            Returns
            -------
            ndarray
                Returns an array of length N, where the i-th element contains the
                squared Mahalanobis distance between (mean, covariance) and
                `measurements[i]`.
            """
            mean, covariance = self.project(mean, covariance)
            
            if only_position:
                mean, covariance = mean[:2], covariance[:2, :2]
                measurements = measurements[:, :2]

            cholesky_factor = np.linalg.cholesky(covariance)
            d = measurements - mean
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            
            return squared_maha