import numpy as np
import cv2


class GaborFilterBank:
    def __init__(self):
        self.orientations = [0, 45, 90, 135]
        self.scales = [2, 4, 8]
        self.kernel_size = 31

        self.filters = self._build_filters()

        print(f"[GaborBank] Built {len(self.filters)} filters")
        print(
            f"{len(self.orientations)} orientations x "
            f"{len(self.scales)} scales"
        )

    def _build_filters(self):
        filters = []

        for sigma in self.scales:
            for theta_deg in self.orientations:
                theta = theta_deg * np.pi / 180.0

                lam = 2.0 * sigma

                kernel_cos = cv2.getGaborKernel(
                    ksize=(self.kernel_size, self.kernel_size),
                    sigma=sigma,
                    theta=theta,
                    lambd=lam,
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_64F
                )

                kernel_sin = cv2.getGaborKernel(
                    ksize=(self.kernel_size, self.kernel_size),
                    sigma=sigma,
                    theta=theta,
                    lambd=lam,
                    gamma=0.5,
                    psi=np.pi / 2,
                    ktype=cv2.CV_64F
                )

                filters.append({
                    "cos": kernel_cos,
                    "sin": kernel_sin,
                    "sigma": sigma,
                    "theta_deg": theta_deg,
                    "scale": {
                        2: "small",
                        4: "normal",
                        8: "large"
                    }[sigma]
                })

        return filters

    def apply(self, gray_image):
        responses = []

        for f in self.filters:
            resp_cos = cv2.filter2D(
                gray_image,
                cv2.CV_64F,
                f["cos"]
            )

            resp_sin = cv2.filter2D(
                gray_image,
                cv2.CV_64F,
                f["sin"]
            )

            magnitude = np.sqrt(
                resp_cos**2 + resp_sin**2
            )

            phase = np.arctan2(
                resp_sin,
                resp_cos
            )

            responses.append({
                "magnitude": magnitude,
                "phase": phase,
                "scale": f["scale"],
                "theta_deg": f["theta_deg"],
                "sigma": f["sigma"]
            })

        return responses

    def visualize(self, gray_image, responses):
        """
        Shows all 12 filter responses in a grid.
        3 rows × 4 columns
        """

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            3,
            4,
            figsize=(14, 9)
        )

        fig.suptitle(
            "Gabor Filter Bank Responses\n"
            "Rows = Scale | Columns = Orientation",
            fontsize=12
        )

        scale_names = [
            "small (σ=2)",
            "normal (σ=4)",
            "large (σ=8)"
        ]

        for i, sigma in enumerate(self.scales):
            for j, theta in enumerate(self.orientations):

                resp = next(
                    r for r in responses
                    if r["sigma"] == sigma
                    and r["theta_deg"] == theta
                )

                ax = axes[i][j]

                ax.imshow(
                    resp["magnitude"],
                    cmap="viridis"
                )

                ax.set_title(
                    f"{scale_names[i]}\n{theta}°",
                    fontsize=9
                )

                ax.axis("off")

        plt.tight_layout()
        plt.show()


# ---------------- TEST ----------------

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    cap.release()

    if ret:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            (320, 240)
        )

        gray = gray.astype(
            np.float32
        ) / 255.0

        bank = GaborFilterBank()

        responses = bank.apply(gray)

        print(
            f"\nFilter responses computed: "
            f"{len(responses)}"
        )

        for r in responses:
            print(
                f"scale={r['scale']:6s} | "
                f"theta={r['theta_deg']:3d}° | "
                f"mean_mag={r['magnitude'].mean():.4f} | "
                f"mean_phase={r['phase'].mean():.4f}"
            )

        bank.visualize(gray, responses)

        print("\nSUCCESS — Gabor filter bank working")