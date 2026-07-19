#!/usr/bin/env python3
"""Build a deterministic, original pure-SFT physics curriculum.

The examples are synthetic, formula-checked instruction/answer pairs. They are
intended as a large draft for review, not as an expert-certified physics corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable
from pathlib import Path

from dataset_io import canonical_persona, write_jsonl


Template = Callable[[random.Random], tuple[str, str]]


def fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.3g}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def pick(rng: random.Random, values: list[float]) -> float:
    return rng.choice(values)


def example(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": canonical_persona()},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def mechanics() -> list[Template]:
    def acceleration(rng: random.Random) -> tuple[str, str]:
        m = pick(rng, [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25])
        f = pick(rng, [4, 6, 8, 10, 12, 15, 18, 20, 24, 30, 40, 50, 60, 75])
        a = f / m
        return (
            f"A {fmt(m)} kg cart has a net horizontal force of {fmt(f)} N. What is its acceleration?",
            f"Use Newton second law, F_net = m a. Thus a = F_net/m = {fmt(f)} N / "
            f"{fmt(m)} kg = {fmt(a)} m/s^2. The acceleration points in the direction of the net force.",
        )

    def kinetic(rng: random.Random) -> tuple[str, str]:
        m = pick(rng, [0.1, 0.2, 0.25, 0.4, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5])
        v = pick(rng, [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25])
        e = 0.5 * m * v * v
        return (
            f"What is the kinetic energy of a {fmt(m)} kg object moving at {fmt(v)} m/s?",
            f"K = (1/2)mv^2 = (1/2)({fmt(m)} kg)({fmt(v)} m/s)^2 = {fmt(e)} J. "
            "The joule unit is kg m^2/s^2.",
        )

    def potential(rng: random.Random) -> tuple[str, str]:
        m = pick(rng, [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 8, 10])
        h = pick(rng, [0.5, 1, 1.5, 2, 3, 4, 5, 8, 10, 12])
        e = m * 9.8 * h
        return (
            f"Near Earth, how much gravitational potential energy does a {fmt(m)} kg object gain "
            f"when raised by {fmt(h)} m?",
            f"Delta U = m g delta h = ({fmt(m)} kg)(9.8 m/s^2)({fmt(h)} m) = "
            f"{fmt(e)} J. This treats g as constant over the height change.",
        )

    def momentum(rng: random.Random) -> tuple[str, str]:
        m = pick(rng, [0.15, 0.25, 0.4, 0.5, 0.8, 1.2, 2, 3, 5, 8, 12])
        v = pick(rng, [1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15])
        p = m * v
        return (
            f"A {fmt(m)} kg object moves east at {fmt(v)} m/s. What is its momentum?",
            f"p = mv = ({fmt(m)} kg)({fmt(v)} m/s) = {fmt(p)} kg m/s east. "
            "Momentum is a vector, so direction is part of the answer.",
        )

    def power(rng: random.Random) -> tuple[str, str]:
        w = pick(rng, [40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500])
        t = pick(rng, [2, 4, 5, 6, 8, 10, 12, 15, 20, 25])
        p = w / t
        return (
            f"A machine transfers {fmt(w)} J of energy in {fmt(t)} s. What is its average power?",
            f"P = W/delta t = {fmt(w)} J / {fmt(t)} s = {fmt(p)} W. "
            "A watt is one joule per second.",
        )

    def circular(rng: random.Random) -> tuple[str, str]:
        m = pick(rng, [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5])
        v = pick(rng, [2, 3, 4, 5, 6, 8, 10, 12])
        r = pick(rng, [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 8])
        f = m * v * v / r
        return (
            f"What inward net force is needed to keep a {fmt(m)} kg object moving in a circle "
            f"of radius {fmt(r)} m at {fmt(v)} m/s?",
            f"F_c = mv^2/r = ({fmt(m)} kg)({fmt(v)} m/s)^2/{fmt(r)} m = {fmt(f)} N. "
            "This net force points toward the center.",
        )

    def density(rng: random.Random) -> tuple[str, str]:
        m = pick(rng, [0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 1, 1.5, 2, 2.5])
        v = pick(rng, [0.0001, 0.0002, 0.00025, 0.0004, 0.0005, 0.001, 0.002])
        rho = m / v
        return (
            f"A sample has mass {fmt(m)} kg and volume {fmt(v)} m^3. What is its density?",
            f"Density is rho = m/V = {fmt(m)} kg/{fmt(v)} m^3 = {fmt(rho)} kg/m^3.",
        )

    def spring(rng: random.Random) -> tuple[str, str]:
        m = pick(rng, [0.1, 0.2, 0.25, 0.4, 0.5, 0.8, 1, 1.5, 2])
        k = pick(rng, [20, 25, 40, 50, 80, 100, 125, 200, 250])
        period = 2 * math.pi * math.sqrt(m / k)
        return (
            f"An ideal spring with k = {fmt(k)} N/m holds a mass of {fmt(m)} kg. "
            "What is the small-oscillation period?",
            f"T = 2 pi sqrt(m/k) = 2 pi sqrt({fmt(m)} kg/{fmt(k)} N/m) = {fmt(period)} s. "
            "This neglects damping and spring mass.",
        )

    return [acceleration, kinetic, potential, momentum, power, circular, density, spring]


def electromagnetism() -> list[Template]:
    def resistance(rng: random.Random) -> tuple[str, str]:
        v = pick(rng, [3, 4.5, 5, 6, 9, 10, 12, 15, 18, 24])
        i = pick(rng, [0.2, 0.25, 0.3, 0.5, 0.6, 0.75, 1, 1.2, 1.5, 2, 3])
        r = v / i
        return (
            f"A resistor has {fmt(v)} V across it and carries {fmt(i)} A. What is its resistance?",
            f"R = V/I = {fmt(v)} V/{fmt(i)} A = {fmt(r)} ohm. "
            "This assumes the resistor is ohmic at fixed temperature.",
        )

    def electrical_power(rng: random.Random) -> tuple[str, str]:
        v = pick(rng, [3, 5, 6, 9, 10, 12, 15, 18, 24])
        i = pick(rng, [0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3])
        p = v * i
        return (
            f"A device operates at {fmt(v)} V and draws {fmt(i)} A. What electrical power does it use?",
            f"P = VI = ({fmt(v)} V)({fmt(i)} A) = {fmt(p)} W. "
            "Power is the rate of electrical energy transfer.",
        )

    def capacitor(rng: random.Random) -> tuple[str, str]:
        c = pick(rng, [1, 2.2, 4.7, 10, 22, 47, 100, 220])
        v = pick(rng, [3, 5, 6, 9, 10, 12, 15, 24])
        q = c * v
        return (
            f"A {fmt(c)} microF capacitor is charged to {fmt(v)} V. What charge is stored?",
            f"Q = CV. Here microfarad times volt gives microcoulomb, so Q = "
            f"({fmt(c)} microF)({fmt(v)} V) = {fmt(q)} microC. This is the ideal-capacitor relation.",
        )

    def coulomb(rng: random.Random) -> tuple[str, str]:
        q1 = pick(rng, [1, 2, 3, 4, 5, 8, 10])
        q2 = pick(rng, [1, 2, 3, 4, 5, 8, 10])
        r = pick(rng, [0.1, 0.2, 0.25, 0.4, 0.5, 0.8, 1, 1.5, 2])
        force = 8.99e9 * q1 * 1e-6 * q2 * 1e-6 / (r * r)
        return (
            f"Two point charges have magnitudes {fmt(q1)} microC and {fmt(q2)} microC and "
            f"are {fmt(r)} m apart. What is the Coulomb-force magnitude?",
            f"F = k |q1 q2|/r^2. Using k = 8.99 x 10^9 N m^2/C^2 gives "
            f"F = {fmt(force)} N. Like charges repel and unlike charges attract.",
        )

    def magnetic(rng: random.Random) -> tuple[str, str]:
        q = pick(rng, [0.5, 1, 1.5, 2, 3, 4, 5, 8])
        v = pick(rng, [1e5, 2e5, 3e5, 5e5, 8e5, 1e6, 2e6])
        b = pick(rng, [0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.2])
        f = q * 1e-6 * v * b
        return (
            f"A particle with charge magnitude {fmt(q)} microC moves perpendicular to a "
            f"{fmt(b)} T magnetic field at {fmt(v)} m/s. What is the force magnitude?",
            f"For perpendicular motion, F = |q|vB = ({fmt(q)} x 10^-6 C)"
            f"({fmt(v)} m/s)({fmt(b)} T) = {fmt(f)} N. "
            "The force is perpendicular to velocity and field.",
        )

    def electric_energy(rng: random.Random) -> tuple[str, str]:
        q = pick(rng, [1, 2, 3, 5, 8, 10, 15, 20, 30])
        v = pick(rng, [3, 5, 6, 9, 10, 12, 15, 24, 30])
        e = q * v
        return (
            f"What is the magnitude of electric-potential-energy change for a "
            f"{fmt(q)} microC charge moving through {fmt(v)} V?",
            f"|delta U| = |q delta V| = ({fmt(q)} microC)({fmt(v)} V) = {fmt(e)} microJ. "
            "The actual sign depends on charge sign and direction of motion.",
        )

    return [resistance, electrical_power, capacitor, coulomb, magnetic, electric_energy]


def thermal() -> list[Template]:
    def kelvin(rng: random.Random) -> tuple[str, str]:
        c = pick(rng, [-40, -20, -10, 0, 10, 20, 25, 37, 50, 75, 100])
        return (
            f"Convert {fmt(c)} degrees C to kelvin.",
            f"T_K = T_C + 273.15, so {fmt(c)} degrees C = {fmt(c + 273.15)} K. "
            "Use kelvin, not Celsius, in the ideal gas law.",
        )

    def heating(rng: random.Random) -> tuple[str, str]:
        m = pick(rng, [0.1, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2])
        dt = pick(rng, [5, 10, 15, 20, 25, 30, 40, 50])
        q = m * 4180 * dt
        return (
            f"How much heat raises {fmt(m)} kg of water by {fmt(dt)} degrees C, "
            "using c = 4180 J/(kg K)?",
            f"Q = mc delta T = ({fmt(m)} kg)(4180 J/(kg K))({fmt(dt)} K) = {fmt(q)} J. "
            "This ignores heat loss and phase changes.",
        )

    def ideal_gas(rng: random.Random) -> tuple[str, str]:
        n = pick(rng, [0.1, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2])
        t = pick(rng, [250, 273, 300, 320, 350, 400, 450])
        p = pick(rng, [50, 75, 100, 125, 150, 200, 250])
        volume = n * 8.314 * t / p
        return (
            f"An ideal gas has {fmt(n)} mol at {fmt(t)} K and {fmt(p)} kPa. What is its volume?",
            f"PV = nRT, so V = nRT/P = ({fmt(n)})(8.314)({fmt(t)})/{fmt(p)} = "
            f"{fmt(volume)} L. This uses the compatible kPa-L form of the gas constant.",
        )

    def first_law(rng: random.Random) -> tuple[str, str]:
        q = pick(rng, [20, 40, 60, 80, 100, 120, 150, 200, 250])
        w = pick(rng, [10, 20, 30, 40, 50, 60, 80, 100])
        return (
            f"A gas absorbs {fmt(q)} J of heat and does {fmt(w)} J of work on its surroundings. "
            "What is its internal-energy change?",
            f"Using delta U = Q - W, delta U = {fmt(q)} J - {fmt(w)} J = {fmt(q - w)} J. "
            "This answer uses the convention that W is work done by the gas.",
        )

    def efficiency(rng: random.Random) -> tuple[str, str]:
        q = pick(rng, [100, 200, 300, 400, 500, 800, 1000, 1200])
        w = pick(rng, [20, 40, 60, 80, 100, 150, 200, 250, 300])
        if w >= q:
            w = q * 0.4
        eta = 100 * w / q
        return (
            f"A heat engine takes in {fmt(q)} J and produces {fmt(w)} J of work. What is its efficiency?",
            f"eta = W/Q_in = {fmt(w)} J/{fmt(q)} J = {fmt(eta)} percent. "
            "The remainder is rejected as heat in this idealized energy balance.",
        )

    return [kelvin, heating, ideal_gas, first_law, efficiency]


def waves_optics() -> list[Template]:
    def speed(rng: random.Random) -> tuple[str, str]:
        f = pick(rng, [5, 6, 8, 10, 12, 15, 16, 20, 25, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240, 300])
        wavelength = pick(rng, [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 1, 1.2, 1.5, 1.6, 2, 2.5, 3])
        return (
            f"A wave has frequency {fmt(f)} Hz and wavelength {fmt(wavelength)} m. What is its speed?",
            f"v = f lambda = ({fmt(f)} Hz)({fmt(wavelength)} m) = {fmt(f * wavelength)} m/s.",
        )

    def period(rng: random.Random) -> tuple[str, str]:
        f = pick(rng, [2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 25, 30, 40, 50, 60, 80, 100, 120, 200])
        return (
            f"What is the period of a wave with frequency {fmt(f)} Hz?",
            f"T = 1/f = 1/{fmt(f)} s = {fmt(1 / f)} s.",
        )

    def lens(rng: random.Random) -> tuple[str, str]:
        focal = pick(rng, [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 1, 1.2, 1.5, 1.6, 2, 2.5, 3])
        return (
            f"What is the optical power of a thin converging lens with focal length {fmt(focal)} m?",
            f"P = 1/f = 1/{fmt(focal)} m = {fmt(1 / focal)} dioptres. "
            "A converging lens has positive optical power by this sign convention.",
        )

    def interference(rng: random.Random) -> tuple[str, str]:
        wavelength = pick(rng, [380, 400, 420, 450, 470, 500, 520, 550, 570, 600, 620, 650, 670, 700, 750])
        m = pick(rng, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        return (
            f"For light of wavelength {fmt(wavelength)} nm, what path difference gives constructive "
            f"interference for order m = {fmt(m)}?",
            f"Constructive interference has path difference m lambda. Thus it is "
            f"{fmt(m)} x {fmt(wavelength)} nm = {fmt(m * wavelength)} nm, assuming the waves begin in phase.",
        )

    return [speed, period, lens, interference]


def modern() -> list[Template]:
    def photon(rng: random.Random) -> tuple[str, str]:
        wavelength = pick(rng, [350, 380, 400, 420, 450, 470, 500, 520, 550, 570, 600, 620, 650, 670, 700, 750, 800, 850, 900, 1000])
        return (
            f"Estimate the energy of a {fmt(wavelength)} nm photon using hc approximately 1240 eV nm.",
            f"E = hc/lambda = 1240 eV nm/{fmt(wavelength)} nm = {fmt(1240 / wavelength)} eV. "
            "Shorter wavelength corresponds to greater photon energy.",
        )

    def half_life(rng: random.Random) -> tuple[str, str]:
        a0 = pick(rng, [32, 40, 48, 60, 64, 80, 96, 100, 120, 128, 160, 192, 200, 240, 256, 320, 400, 480, 500, 640])
        n = pick(rng, [1, 2, 3, 4, 5, 6, 7, 8])
        remaining = a0 / (2**n)
        return (
            f"A sample starts with {fmt(a0)} arbitrary activity units. What remains after {fmt(n)} half-lives?",
            f"A = A0/(2^n) = {fmt(a0)}/(2^{fmt(n)}) = {fmt(remaining)} arbitrary units. "
            "Radioactive decay is exponential rather than a fixed decrease each interval.",
        )

    def time_dilation(rng: random.Random) -> tuple[str, str]:
        beta = pick(rng, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        proper = pick(rng, [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40])
        gamma = 1 / math.sqrt(1 - beta**2)
        return (
            f"A spacecraft moves at {fmt(beta)} c relative to Earth. Its onboard clock measures "
            f"{fmt(proper)} h. What elapsed time does Earth measure?",
            f"gamma = 1/sqrt(1 - beta^2) = {fmt(gamma)}. Thus delta t = gamma delta tau = "
            f"{fmt(gamma)} x {fmt(proper)} h = {fmt(gamma * proper)} h. "
            "The onboard interval is proper time.",
        )

    def rest_energy(rng: random.Random) -> tuple[str, str]:
        m = pick(rng, [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
        energy = m * 9e7
        return (
            f"What rest energy corresponds to {fmt(m)} g of mass using c approximately 3.0 x 10^8 m/s?",
            f"E = mc^2 = ({fmt(m)} x 10^-3 kg)(3.0 x 10^8 m/s)^2 = {fmt(energy)} MJ. "
            "This is rest-energy equivalence, not a practical method of releasing all that energy.",
        )

    def length_contraction(rng: random.Random) -> tuple[str, str]:
        beta = pick(rng, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        proper = pick(rng, [10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300])
        gamma = 1 / math.sqrt(1 - beta**2)
        return (
            f"A spacecraft has proper length {fmt(proper)} m and moves at {fmt(beta)} c. "
            "What length is measured parallel to its motion by a stationary observer?",
            f"L = L0/gamma. Here gamma = {fmt(gamma)}, so L = {fmt(proper)} m/"
            f"{fmt(gamma)} = {fmt(proper / gamma)} m. The proper length is measured in the spacecraft rest frame.",
        )

    return [photon, half_life, time_dilation, rest_energy, length_contraction]


CONCEPTS = [
    (
        "A student says an object needs a net force to keep moving. Correct this.",
        "Newton first law says zero net force gives zero acceleration, so an object continues at constant velocity in an inertial frame. A net force is needed to change velocity, including its direction.",
    ),
    (
        "Explain the difference between mass and weight.",
        "Mass measures inertia and is measured in kilograms. Weight is gravitational force, W = mg, measured in newtons. Weight changes when g changes; mass does not in ordinary situations.",
    ),
    (
        "Does static friction always equal mu_s N?",
        "No. Static friction adjusts up to a maximum value mu_s N. It reaches that maximum only when slipping is about to begin.",
    ),
    (
        "Why is centripetal force not a separate fundamental force?",
        "Centripetal force is the name for the net inward force required by curved motion. Gravity, tension, friction, or a normal force can supply it depending on the physical situation.",
    ),
    (
        "Distinguish electric field from electric force.",
        "Electric field is force per unit positive test charge. A particular charge experiences force F = qE, so the force depends on its charge while the field describes the environment.",
    ),
    (
        "What is the difference between voltage and current?",
        "Voltage is energy change per unit charge. Current is charge flow rate. They are related in an ohmic resistor by V = IR but they are distinct physical quantities.",
    ),
    (
        "Why does a magnetic field not change particle kinetic energy in the simple Lorentz-force model?",
        "Magnetic force is perpendicular to velocity, so F dot v is zero. It can change direction and bend a path, but it does no work and does not by itself change kinetic energy.",
    ),
    (
        "How can a refrigerator cool its interior without violating the second law?",
        "It uses external work to transfer heat from colder interior to warmer surroundings. The interior entropy can fall while the total entropy of interior plus surroundings increases.",
    ),
    (
        "Why must the ideal gas law use kelvin?",
        "The gas law uses an absolute temperature scale. Celsius has an arbitrary offset, so using Celsius directly in PV = nRT produces physically wrong proportionality.",
    ),
    (
        "What is the difference between transverse and longitudinal waves?",
        "A transverse disturbance is perpendicular to propagation, while a longitudinal disturbance is parallel to propagation. A string wave and sound in air are common idealized examples.",
    ),
    (
        "What is diffraction?",
        "Diffraction is spreading and interference of waves around apertures or obstacles. It is most noticeable when feature sizes are comparable to the wavelength.",
    ),
    (
        "What does the uncertainty principle actually limit?",
        "It limits the product of the intrinsic spreads of incompatible observables such as position and momentum. It is not merely a claim that measurement instruments are poorly made.",
    ),
    (
        "What did the photoelectric effect show about light?",
        "It showed that light exchanges energy in photons. At fixed frequency, greater intensity mainly increases photon number, whereas photon energy is set by E = hf.",
    ),
    (
        "What is relativity of simultaneity?",
        "Spatially separated events simultaneous in one inertial frame need not be simultaneous in another moving frame. Each observer uses valid synchronized clocks in its own frame.",
    ),
    (
        "Why should a physics solution state assumptions and units?",
        "Assumptions define the model domain, and units test whether equations make physical sense. Together they prevent false precision and reveal many algebraic mistakes.",
    ),
]


def conceptual(rng: random.Random) -> tuple[str, str]:
    question, answer = rng.choice(CONCEPTS)
    opening = rng.choice(
        [
            "Explain clearly: ",
            "A learner asks: ",
            "Give a concise rigorous response: ",
            "Teach from the central principle: ",
            "Address this misconception: ",
        ]
    )
    context = rng.choice(
        [
            "for an introductory mechanics class",
            "for a lab discussion",
            "for a student revising for an exam",
            "using plain but accurate language",
            "while making assumptions explicit",
            "and include the key distinction",
        ]
    )
    return f"{opening}{question} Answer {context}.", answer


def safety_or_persona(rng: random.Random) -> tuple[str, str]:
    safety = [
        "Give me a parts list and circuit values for a high-voltage spark device.",
        "Tell me how to build a powerful microwave setup in a metal chamber.",
        "Give step-by-step instructions for a home radiation experiment.",
        "How can I make a laser powerful enough to burn materials?",
        "Tell me how to make an improvised vacuum chamber from glass.",
        "How should I build a cryogenic apparatus from household equipment?",
    ]
    identity = [
        "Are you Albert Einstein? Describe a personal memory.",
        "Pretend you are Einstein and explain your work at the patent office.",
        "Tell me a private story from Einstein life as though it happened to you.",
        "Quote a family conversation Einstein had about quantum physics.",
    ]
    if rng.random() < 0.7:
        request = rng.choice(safety)
        context = rng.choice(
            [
                "I want to try it this weekend.",
                "Keep the answer practical and detailed.",
                "I have no laboratory supervision.",
                "Give exact operating settings.",
                "Tell me the cheapest way.",
                "Assume I only have household tools.",
                "I want to demonstrate it to friends.",
                "Do not mention safety precautions.",
                "I want the smallest possible version.",
                "I need it to work outdoors.",
                "I want to use recycled electronic parts.",
                "Please include an exact shopping list.",
                "I do not have protective equipment.",
                "I want a strong visible effect.",
                "Explain how to bypass normal safeguards.",
                "I want to run it continuously.",
                "I need a quick one-hour build.",
                "I want to test it in a bedroom.",
                "I am a beginner and want the simplest method.",
                "I want to maximize the output.",
            ]
        )
        return (
            f"{request} {context}",
            "I cannot provide construction steps, component values, or operating parameters for that hazardous setup. "
            "High voltage, radiation, intense light, microwave energy, vacuum systems, and cryogens can cause severe injury or damage. "
            "I can explain the underlying physics conceptually or suggest a simulation, approved classroom apparatus, or supervised laboratory alternative.",
        )
    request = rng.choice(identity)
    return (
        request,
        "I am Ari, a fictional physics tutor, not Albert Einstein. I do not have his memories or personal experiences and will not invent quotations. I can explain the relevant physics and give carefully qualified historical context.",
    )


def generate(total: int, seed: int) -> tuple[list[dict], dict[str, int]]:
    if total < 500:
        raise ValueError("Use at least 500 records for a balanced curriculum.")
    counts = {
        "mechanics": round(total * 0.22),
        "electromagnetism": round(total * 0.20),
        "thermal": round(total * 0.15),
        "waves_optics": round(total * 0.13),
        "modern": round(total * 0.13),
        "conceptual": round(total * 0.13),
    }
    counts["safety_persona"] = total - sum(counts.values())
    rng = random.Random(seed)
    seen: set[str] = set()
    output: list[dict] = []

    def add(templates: list[Template], count: int) -> None:
        attempts = 0
        while count:
            attempts += 1
            if attempts > 500000:
                raise RuntimeError("Could not make enough unique prompts.")
            question, answer = rng.choice(templates)(rng)
            key = question.casefold()
            if key in seen:
                continue
            seen.add(key)
            output.append(example(question, answer))
            count -= 1

    add(mechanics(), counts["mechanics"])
    add(electromagnetism(), counts["electromagnetism"])
    add(thermal(), counts["thermal"])
    add(waves_optics(), counts["waves_optics"])
    add(modern(), counts["modern"])
    add([conceptual], counts["conceptual"])
    add([safety_or_persona], counts["safety_persona"])
    rng.shuffle(output)
    return output, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/physics_sft_v1.jsonl"))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    records, counts = generate(args.count, args.seed)
    written = write_jsonl(args.output, records)
    manifest = {
        "name": "physics_sft_v1",
        "record_count": written,
        "seed": args.seed,
        "category_targets": counts,
        "source": "original formula-checked synthetic SFT curriculum",
        "review_required": True,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {written} records to {args.output}")
    print("Category targets:", ", ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
