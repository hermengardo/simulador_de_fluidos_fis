// ----- ENUMS and CONSTANTS -----
    const U_FIELD = 0;
    const V_FIELD = 1;
    const S_FIELD = 2;

    /**
     * Represents the main fluid simulation.
     * Based on the work of Jos Stam, "Real-Time Fluid Dynamics for Games".
     * And the simplified implementation by Matthias MÃƒÂ¼ller.
     */
    class Fluid {
        // Properties
        density;
        numX;
        numY;
        numCells;
        h;
        u;
        v;
        newU;
        newV;
        p;
        s;
        m;
        newM;

        constructor(density, numX, numY, h) {
            this.density = density;
            this.numX = numX + 2;
            this.numY = numY + 2;
            this.numCells = this.numX * this.numY;
            this.h = h;

            // Velocities
            this.u = new Float32Array(this.numCells);
            this.v = new Float32Array(this.numCells);
            this.newU = new Float32Array(this.numCells);
            this.newV = new Float32Array(this.numCells);

            // Pressure
            this.p = new Float32Array(this.numCells);

            // Obstacles/Solids
            this.s = new Float32Array(this.numCells);

            // Smoke
            this.m = new Float32Array(this.numCells);
            this.newM = new Float32Array(this.numCells);

            this.m.fill(1.0);
        }

        /**
         * Integrates external forces like gravity.
         * @param {number} dt - Time step.
         * @param {number} gravity - Gravity constant.
         */
        integrate(dt, gravity) {
            const n = this.numY;
            for (let i = 1; i < this.numX; i++) {
                for (let j = 1; j < this.numY - 1; j++) {
                    if (this.s[i * n + j] !== 0.0 && this.s[i * n + j - 1] !== 0.0) {
                        this.v[i * n + j] += gravity * dt;
                    }
                }
            }
        }

        /**
         * Enforces the incompressibility condition using projection.
         * This step ensures the fluid doesn't "disappear" or "get created" out of nowhere.
         * It uses a Gauss-Seidel relaxation method.
         * @param {number} numIters - Number of iterations for the solver.
         * @param {number} dt - Time step.
         */
        solveIncompressibility(numIters, dt, overRelaxation) {
            const n = this.numY;
            const cp = this.density * this.h / dt;

            for (let iter = 0; iter < numIters; iter++) {
                for (let i = 1; i < this.numX - 1; i++) {
                    for (let j = 1; j < this.numY - 1; j++) {
                        if (this.s[i * n + j] === 0.0) continue;

                        const sx0 = this.s[(i - 1) * n + j];
                        const sx1 = this.s[(i + 1) * n + j];
                        const sy0 = this.s[i * n + j - 1];
                        const sy1 = this.s[i * n + j + 1];
                        const s = sx0 + sx1 + sy0 + sy1;
                        if (s === 0.0) continue;

                        const div = this.u[(i + 1) * n + j] - this.u[i * n + j] +
                                    this.v[i * n + j + 1] - this.v[i * n + j];

                        let p = -div / s;
                        p *= overRelaxation;
                        this.p[i * n + j] += cp * p;

                        this.u[i * n + j] -= sx0 * p;
                        this.u[(i + 1) * n + j] += sx1 * p;
                        this.v[i * n + j] -= sy0 * p;
                        this.v[i * n + j + 1] += sy1 * p;
                    }
                }
            }
        }

        /**
         * Extrapolates velocities into solid cells for boundary conditions.
         */
        extrapolate() {
            const n = this.numY;
            for (let i = 0; i < this.numX; i++) {
                this.u[i * n + 0] = this.u[i * n + 1];
                this.u[i * n + this.numY - 1] = this.u[i * n + this.numY - 2];
            }
            for (let j = 0; j < this.numY; j++) {
                this.v[0 * n + j] = this.v[1 * n + j];
                this.v[(this.numX - 1) * n + j] = this.v[(this.numX - 2) * n + j];
            }
        }

        /**
         * Samples a field at a given position using bilinear interpolation.
         * @param {number} x - x-coordinate.
         * @param {number} y - y-coordinate.
         * @param {number} field - The field to sample (U_FIELD, V_FIELD, S_FIELD).
         * @returns {number} The interpolated value.
         */
        sampleField(x, y, field) {
            const n = this.numY;
            const h = this.h;
            const h1 = 1.0 / h;
            const h2 = 0.5 * h;

            x = Math.max(Math.min(x, this.numX * h), h);
            y = Math.max(Math.min(y, this.numY * h), h);

            let dx = 0.0;
            let dy = 0.0;
            let f;

            switch (field) {
                case U_FIELD: f = this.u; dy = h2; break;
                case V_FIELD: f = this.v; dx = h2; break;
                case S_FIELD: f = this.m; dx = h2; dy = h2; break;
                default: return 0;
            }

            const x0 = Math.min(Math.floor((x - dx) * h1), this.numX - 1);
            const tx = ((x - dx) - x0 * h) * h1;
            const x1 = Math.min(x0 + 1, this.numX - 1);

            const y0 = Math.min(Math.floor((y - dy) * h1), this.numY - 1);
            const ty = ((y - dy) - y0 * h) * h1;
            const y1 = Math.min(y0 + 1, this.numY - 1);

            const sx = 1.0 - tx;
            const sy = 1.0 - ty;

            return sx * sy * f[x0 * n + y0] +
                   tx * sy * f[x1 * n + y0] +
                   tx * ty * f[x1 * n + y1] +
                   sx * ty * f[x0 * n + y1];
        }

        avgU(i, j) {
            const n = this.numY;
            return (this.u[i * n + j - 1] + this.u[i * n + j] +
                    this.u[(i + 1) * n + j - 1] + this.u[(i + 1) * n + j]) * 0.25;
        }

        avgV(i, j) {
            const n = this.numY;
            return (this.v[(i - 1) * n + j] + this.v[i * n + j] +
                    this.v[(i - 1) * n + j + 1] + this.v[i * n + j + 1]) * 0.25;
        }

        /**
         * Advects the velocity field. This means moving the velocities themselves along the flow.
         * @param {number} dt - Time step.
         */
        advectVel(dt) {
            this.newU.set(this.u);
            this.newV.set(this.v);

            const n = this.numY;
            const h = this.h;
            const h2 = 0.5 * h;

            for (let i = 1; i < this.numX; i++) {
                for (let j = 1; j < this.numY; j++) {
                    // u component
                    if (this.s[i * n + j] !== 0.0 && this.s[(i - 1) * n + j] !== 0.0 && j < this.numY - 1) {
                        let x = i * h;
                        let y = j * h + h2;
                        let u = this.u[i * n + j];
                        let v = this.avgV(i, j);
                        x -= dt * u;
                        y -= dt * v;
                        this.newU[i * n + j] = this.sampleField(x, y, U_FIELD);
                    }
                    // v component
                    if (this.s[i * n + j] !== 0.0 && this.s[i * n + j - 1] !== 0.0 && i < this.numX - 1) {
                        let x = i * h + h2;
                        let y = j * h;
                        let u = this.avgU(i, j);
                        let v = this.v[i * n + j];
                        x -= dt * u;
                        y -= dt * v;
                        this.newV[i * n + j] = this.sampleField(x, y, V_FIELD);
                    }
                }
            }
            this.u.set(this.newU);
            this.v.set(this.newV);
        }

        /**
         * Advects the smoke (or any scalar quantity) field.
         * @param {number} dt - Time step.
         */
        advectSmoke(dt) {
            this.newM.set(this.m);
            const n = this.numY;
            const h = this.h;
            const h2 = 0.5 * h;

            for (let i = 1; i < this.numX - 1; i++) {
                for (let j = 1; j < this.numY - 1; j++) {
                    if (this.s[i * n + j] !== 0.0) {
                        const u = (this.u[i * n + j] + this.u[(i + 1) * n + j]) * 0.5;
                        const v = (this.v[i * n + j] + this.v[i * n + j + 1]) * 0.5;
                        const x = i * h + h2 - dt * u;
                        const y = j * h + h2 - dt * v;
                        this.newM[i * n + j] = this.sampleField(x, y, S_FIELD);
                    }
                }
            }
            this.m.set(this.newM);
        }

        /**
         * The main simulation step.
         * @param {Scene} scene - The scene object containing simulation parameters.
         */
        simulate(scene) {
            this.integrate(scene.dt, scene.gravity);
            this.p.fill(0.0);
            this.solveIncompressibility(scene.numIters, scene.dt, scene.overRelaxation);
            this.extrapolate();
            this.advectVel(scene.dt);
            this.advectSmoke(scene.dt);
        }
    }

    /**
     * Manages the simulation scene, parameters, and rendering.
     */
    class Scene {
        // Properties
        canvas;
        c;
        simHeight = 1.1;
        cScale;
        simWidth;
        gravity = -9.81;
        dt = 1.0 / 120.0;
        numIters = 100;
        frameNr = 0;
        overRelaxation = 1.9;
        obstacleX = 0.0;
        obstacleY = 0.0;
        obstacleRadius = 0.15;
        paused = false;
        sceneNr = 0;
        showObstacle = false;
        showStreamlines = false;
        showVelocities = false;
        showPressure = false;
        showSmoke = true;
        fluid = null;
        mouseDown = false;

        constructor() {
            this.canvas = document.getElementById('fluidCanvas');
            this.c = this.canvas.getContext('2d');
            this.updateCanvasSize();
            this.setupEventListeners();
        }

        updateCanvasSize() {
            const container = document.querySelector('.container');
            const containerStyles = getComputedStyle(container);
            const containerWidth = container.clientWidth
                - parseFloat(containerStyles.paddingLeft)
                - parseFloat(containerStyles.paddingRight);

            this.canvas.width = Math.min(containerWidth, window.innerWidth - 40);
            this.canvas.height = Math.min(600, window.innerHeight - 250);

            this.cScale = this.canvas.height / this.simHeight;
            this.simWidth = this.canvas.width / this.cScale;
        }

        setupEventListeners() {
            document.getElementById('windTunnel').addEventListener('click', () => this.setupScene(1));
            document.getElementById('hiresTunnel').addEventListener('click', () => this.setupScene(3));
            document.getElementById('tank').addEventListener('click', () => this.setupScene(0));
            document.getElementById('paint').addEventListener('click', () => this.setupScene(2));

            const streamButton = document.getElementById('streamButton');
            streamButton.addEventListener('click', () => this.showStreamlines = streamButton.checked);

            const velocityButton = document.getElementById('velocityButton');
            velocityButton.addEventListener('click', () => this.showVelocities = velocityButton.checked);

            const pressureButton = document.getElementById('pressureButton');
            pressureButton.addEventListener('click', () => this.showPressure = pressureButton.checked);

            const smokeButton = document.getElementById('smokeButton');
            smokeButton.addEventListener('click', () => this.showSmoke = smokeButton.checked);

            const overrelaxButton = document.getElementById('overrelaxButton');
            overrelaxButton.addEventListener('click', () => this.overRelaxation = overrelaxButton.checked ? 1.9 : 1.0);

            // Mouse and Touch events
            this.canvas.addEventListener('mousedown', e => this.startDrag(e.clientX, e.clientY));
            this.canvas.addEventListener('mouseup', () => this.endDrag());
            this.canvas.addEventListener('mousemove', e => this.drag(e.clientX, e.clientY));

            this.canvas.addEventListener('touchstart', e => this.startDrag(e.touches[0].clientX, e.touches[0].clientY));
            this.canvas.addEventListener('touchend', () => this.endDrag());
            this.canvas.addEventListener('touchmove', e => {
                e.preventDefault();
                e.stopImmediatePropagation();
                this.drag(e.touches[0].clientX, e.touches[0].clientY);
            }, { passive: false });

            document.addEventListener('keydown', e => {
                if (e.key === 'p') this.paused = !this.paused;
            });

            window.addEventListener('resize', () => {
                this.updateCanvasSize();
                this.setupScene(this.sceneNr);
            });
        }

        /**
         * Sets up a specific simulation scenario.
         * @param {number} sceneNr - The ID of the scene to set up.
         */
        setupScene(sceneNr = 0) {
            this.sceneNr = sceneNr;
            this.obstacleRadius = 0.15;
            this.overRelaxation = 1.9;
            this.dt = 1.0 / 60.0;
            this.numIters = 40;

            let res = 100;
            if (sceneNr === 0) res = 50;
            else if (sceneNr === 3) res = 100;

            const domainHeight = 1.0;
            const domainWidth = domainHeight / this.simHeight * this.simWidth;
            const h = domainHeight / res;

            const numX = Math.floor(domainWidth / h);
            const numY = Math.floor(domainHeight / h);
            const density = 1000.0;

            this.fluid = new Fluid(density, numX, numY, h);
            const f = this.fluid;
            const n = f.numY;

            // Scene-specific setups
            switch(sceneNr) {
                case 0: // Tank
                    for (let i = 0; i < f.numX; i++) {
                        for (let j = 0; j < f.numY; j++) {
                            f.s[i * n + j] = (i === 0 || i === f.numX - 1 || j === 0) ? 0.0 : 1.0;
                        }
                    }
                    this.gravity = -9.81;
                    this.showPressure = true;
                    this.showSmoke = false;
                    this.showStreamlines = false;
                    this.showVelocities = false;
                    this.showObstacle = false;
                    break;

                case 1: // Wind Tunnel
                case 3: // Hires Wind Tunnel
                    const inVel = 2.0;
                    for (let i = 0; i < f.numX; i++) {
                        for (let j = 0; j < f.numY; j++) {
                            f.s[i * n + j] = (i === 0 || j === 0 || j === f.numY - 1) ? 0.0 : 1.0;
                            if (i === 1) f.u[i * n + j] = inVel;
                        }
                    }
                    const pipeH = 0.1 * f.numY;
                    const minJ = Math.floor(0.5 * f.numY - 0.5 * pipeH);
                    const maxJ = Math.floor(0.5 * f.numY + 0.5 * pipeH);

                    for (let j = minJ; j < maxJ; j++) {
                         for (let i = 0; i < f.numX; i++) {
                            if (i < 5) f.m[i*n + j] = 0.0;
                         }
                    }

                    this.setObstacle(0.4, 0.5, true);
                    this.gravity = 0.0;
                    this.showSmoke = true;
                    this.showStreamlines = false;
                    this.showVelocities = false;
                    this.showPressure = sceneNr === 3;
                    if (sceneNr === 3) {
                       this.dt = 1.0 / 120.0;
                       this.numIters = 100;
                    }
                    break;

                case 2: // Paint
                    this.gravity = 0.0;
                    this.overRelaxation = 1.0;
                    this.showPressure = false;
                    this.showSmoke = true;
                    this.showStreamlines = false;
                    this.showVelocities = false;
                    this.obstacleRadius = 0.1;
                    this.showObstacle = true;
                     for (let i = 0; i < f.numX; i++) {
                        for (let j = 0; j < f.numY; j++) {
                            f.s[i * n + j] = (i === 0 || i === f.numX - 1 || j === 0 || j === f.numY-1) ? 0.0 : 1.0;
                        }
                    }
                    break;
            }

            // Update UI to match scene state
            document.getElementById('streamButton').checked = this.showStreamlines;
            document.getElementById('velocityButton').checked = this.showVelocities;
            document.getElementById('pressureButton').checked = this.showPressure;
            document.getElementById('smokeButton').checked = this.showSmoke;
            document.getElementById('overrelaxButton').checked = this.overRelaxation > 1.0;
        }

        // --- Drawing Methods ---

        cX(x) { return x * this.cScale; }
        cY(y) { return this.canvas.height - y * this.cScale; }

        getSciColor(val, minVal, maxVal) {
            val = Math.min(Math.max(val, minVal), maxVal - 0.0001);
            const d = maxVal - minVal;
            val = d === 0.0 ? 0.5 : (val - minVal) / d;
            const m = 0.25;
            const num = Math.floor(val / m);
            const s = (val - num * m) / m;
            let r, g, b;

            switch (num) {
                case 0: r = 0.0; g = s; b = 1.0; break;
                case 1: r = 0.0; g = 1.0; b = 1.0 - s; break;
                case 2: r = s; g = 1.0; b = 0.0; break;
                case 3: r = 1.0; g = 1.0 - s; b = 0.0; break;
                default: r=1.0; g=1.0; b=1.0; break;
            }
            return [255 * r, 255 * g, 255 * b];
        }

        draw() {
            if (!this.fluid) return;

            this.c.clearRect(0, 0, this.canvas.width, this.canvas.height);
            const f = this.fluid;
            const n = f.numY;
            const h = f.h;

            let minP = f.p[0], maxP = f.p[0];
            if (this.showPressure) {
                for (let i = 0; i < f.numCells; i++) {
                    minP = Math.min(minP, f.p[i]);
                    maxP = Math.max(maxP, f.p[i]);
                }
            }

            const id = this.c.createImageData(this.canvas.width, this.canvas.height);

            for (let i = 0; i < f.numX; i++) {
                for (let j = 0; j < f.numY; j++) {
                    let color = [255, 255, 255];

                    if (this.showPressure) {
                        const p = f.p[i * n + j];
                        color = this.getSciColor(p, minP, maxP);
                        if (this.showSmoke) {
                            const s = f.m[i * n + j];
                            color[0] = Math.max(0.0, color[0] - 255 * s);
                            color[1] = Math.max(0.0, color[1] - 255 * s);
                            color[2] = Math.max(0.0, color[2] - 255 * s);
                        }
                    } else if (this.showSmoke) {
                        const s = f.m[i * n + j];
                        if (this.sceneNr === 2) {
                             color = this.getSciColor(s, 0.0, 1.0);
                        } else {
                            color = [255 * s, 255 * s, 255 * s];
                        }
                    } else if (f.s[i * n + j] === 0.0) {
                        color = [0, 0, 0];
                    }

                    const x = Math.floor(this.cX(i * h));
                    const y = Math.floor(this.cY((j + 1) * h));
                    const cx = Math.floor(this.cScale * 1.1 * h) + 1;
                    const cy = Math.floor(this.cScale * 1.1 * h) + 1;
                    const [r, g, b] = color;

                    for (let yi = y; yi < y + cy; yi++) {
                        if (yi < 0 || yi >= this.canvas.height) continue;
                        const p = 4 * (yi * this.canvas.width + x);
                        for (let xi = 0; xi < cx; xi++) {
                            if (x + xi < 0 || x + xi >= this.canvas.width) continue;
                            id.data[p + xi * 4] = r;
                            id.data[p + xi * 4 + 1] = g;
                            id.data[p + xi * 4 + 2] = b;
                            id.data[p + xi * 4 + 3] = 255;
                        }
                    }
                }
            }
            this.c.putImageData(id, 0, 0);

            if (this.showVelocities) { this.drawVelocities(); }
            if (this.showStreamlines) { this.drawStreamlines(); }
            if (this.showObstacle) { this.drawObstacle(); }

            if (this.showPressure) {
                const s = `Pressure: ${minP.toFixed(0)} - ${maxP.toFixed(0)} N/m²`;
                this.c.fillStyle = "#000000";
                this.c.font = "16px Arial";
                this.c.fillText(s, 10, 20);
            }
        }

        drawVelocities() {
            const f = this.fluid;
            const n = f.numY;
            const h = f.h;
            this.c.strokeStyle = "#000000";
            const scale = 0.02;

            for (let i = 0; i < f.numX; i++) {
                for (let j = 0; j < f.numY; j++) {
                    const u = f.u[i * n + j];
                    const v = f.v[i * n + j];

                    const x0 = this.cX(i * h);
                    const y = this.cY((j + 0.5) * h);
                    this.c.beginPath();
                    this.c.moveTo(x0, y);
                    this.c.lineTo(this.cX(i * h + u * scale), y);
                    this.c.stroke();

                    const x = this.cX((i + 0.5) * h);
                    const y0 = this.cY(j * h);
                    this.c.beginPath();
                    this.c.moveTo(x, y0);
                    this.c.lineTo(x, this.cY(j * h + v * scale));
                    this.c.stroke();
                }
            }
        }

        drawStreamlines() {
            const f = this.fluid;
            const numSegs = 15;
            this.c.strokeStyle = "#000000";

            for (let i = 1; i < f.numX - 1; i += 5) {
                for (let j = 1; j < f.numY - 1; j += 5) {
                    let x = (i + 0.5) * f.h;
                    let y = (j + 0.5) * f.h;

                    this.c.beginPath();
                    this.c.moveTo(this.cX(x), this.cY(y));

                    for (let n = 0; n < numSegs; n++) {
                        const u = f.sampleField(x, y, U_FIELD);
                        const v = f.sampleField(x, y, V_FIELD);
                        x += u * 0.01;
                        y += v * 0.01;
                        if (x > f.numX * f.h) break;
                        this.c.lineTo(this.cX(x), this.cY(y));
                    }
                    this.c.stroke();
                }
            }
        }

        drawObstacle() {
             const f = this.fluid;
            const r = this.obstacleRadius + f.h;
            this.c.fillStyle = this.showPressure ? "#000000" : "#DDDDDD";
            this.c.beginPath();
            this.c.arc(this.cX(this.obstacleX), this.cY(this.obstacleY), this.cScale * r, 0.0, 2.0 * Math.PI);
            this.c.closePath();
            this.c.fill();

            this.c.lineWidth = 3.0;
            this.c.strokeStyle = "#000000";
            this.c.beginPath();
            this.c.arc(this.cX(this.obstacleX), this.cY(this.obstacleY), this.cScale * r, 0.0, 2.0 * Math.PI);
            this.c.closePath();
            this.c.stroke();
            this.c.lineWidth = 1.0;
        }


        // --- Interaction ---

        setObstacle(x, y, reset) {
            if (!this.fluid) return;

            let vx = 0.0;
            let vy = 0.0;
            if (!reset) {
                vx = (x - this.obstacleX) / this.dt;
                vy = (y - this.obstacleY) / this.dt;
            }

            this.obstacleX = x;
            this.obstacleY = y;
            const r = this.obstacleRadius;
            const f = this.fluid;
            const n = f.numY;

            for (let i = 1; i < f.numX - 1; i++) {
                for (let j = 1; j < f.numY - 1; j++) {
                    f.s[i * n + j] = 1.0;
                    const dx = (i + 0.5) * f.h - x;
                    const dy = (j + 0.5) * f.h - y;

                    if (dx * dx + dy * dy < r * r) {
                        f.s[i * n + j] = 0.0;
                        if (this.sceneNr === 2) {
                            f.m[i * n + j] = 0.5 + 0.5 * Math.sin(0.1 * this.frameNr);
                        } else {
                            f.m[i * n + j] = 1.0;
                        }
                        f.u[i * n + j] = vx;
                        f.u[(i + 1) * n + j] = vx;
                        f.v[i * n + j] = vy;
                        f.v[i * n + j + 1] = vy;
                    }
                }
            }
            this.showObstacle = true;
        }

        startDrag(x, y) {
            const bounds = this.canvas.getBoundingClientRect();
            const mx = x - bounds.left;
            const my = y - bounds.top;
            this.mouseDown = true;

            const simX = mx / this.cScale;
            const simY = (this.canvas.height - my) / this.cScale;
            this.setObstacle(simX, simY, true);
        }

        drag(x, y) {
            if (this.mouseDown) {
                const bounds = this.canvas.getBoundingClientRect();
                const mx = x - bounds.left;
                const my = y - bounds.top;
                const simX = mx / this.cScale;
                const simY = (this.canvas.height - my) / this.cScale;
                this.setObstacle(simX, simY, false);
            }
        }

        endDrag() {
            this.mouseDown = false;
        }

        // --- Main Loop ---

        run() {
            if (!this.paused && this.fluid) {
                this.fluid.simulate(this);
                this.frameNr++;
            }
            this.draw();
            requestAnimationFrame(() => this.run());
        }
    }

    // --- Entry Point ---

    const scene = new Scene();
    scene.setupScene(1);
    scene.run();
