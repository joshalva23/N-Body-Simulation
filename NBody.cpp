#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#define G 6.67430e-11
#define TIMESTEP 5
#define WINDOW_SIZE 800
#define NUM_BODIES 10000
#define MASS_INCREASE 50
#define THREADS 8
#define COLLISION_RADIUS 10.0
#define SOFTENING_FACTOR 1e-4

struct Body {
    double x, y, vx, vy, ax, ay, mass;
    sf::Color color;
};

// Compute gravitational forces
void compute_forces(std::vector<Body>& bodies) {
    int N = bodies.size();

    for (int i = 0; i < N; i++) {
        double fx = 0, fy = 0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double distSq = dx * dx + dy * dy + SOFTENING_FACTOR; // Avoid division by zero
                double invDist = 1.0 / sqrt(distSq);
                double force = G * bodies[i].mass * bodies[j].mass * invDist * invDist;
                fx += force * dx * invDist;
                fy += force * dy * invDist;
            }
        }
        bodies[i].ax = fx / bodies[i].mass;
        bodies[i].ay = fy / bodies[i].mass;
    }
}

// Velocity Verlet Integration
void update_positions(std::vector<Body>& bodies) {

    for (int i = 0; i < bodies.size(); i++) {
        bodies[i].vx += bodies[i].ax * TIMESTEP * 0.5;
        bodies[i].vy += bodies[i].ay * TIMESTEP * 0.5;
        bodies[i].x += bodies[i].vx * TIMESTEP;
        bodies[i].y += bodies[i].vy * TIMESTEP;
    }
}

// Efficient collision handling
void handle_collisions(std::vector<Body>& bodies) {
    int N = bodies.size();
    std::vector<bool> toRemove(N, false);

    for (int i = 0; i < N; i++) {
        if (toRemove[i]) continue;
        for (int j = i + 1; j < N; j++) {
            if (toRemove[j]) continue;
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double distSq = dx * dx + dy * dy;

            if (distSq < COLLISION_RADIUS * COLLISION_RADIUS) {
                double totalMass = bodies[i].mass + bodies[j].mass;
                bodies[i].vx = (bodies[i].vx * bodies[i].mass + bodies[j].vx * bodies[j].mass) / totalMass;
                bodies[i].vy = (bodies[i].vy * bodies[i].mass + bodies[j].vy * bodies[j].mass) / totalMass;
                bodies[i].mass = totalMass;
                bodies[i].color = sf::Color(rand() % 255, rand() % 255, rand() % 255);
                toRemove[j] = true;
            }
        }
    }

    // Remove merged bodies
    bodies.erase(std::remove_if(bodies.begin(), bodies.end(),
                [&](const Body& b) { return toRemove[&b - &bodies[0]]; }), bodies.end());
}

// Check if a body is clicked
int find_clicked_body(const std::vector<Body>& bodies, int mouseX, int mouseY) {
    for (int i = 0; i < bodies.size(); i++) {
        double dx = bodies[i].x - mouseX;
        double dy = bodies[i].y - mouseY;
        if (sqrt(dx * dx + dy * dy) < 8.0) return i;
    }
    return -1;
}

int main() {
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Fast N-Body Simulation");

    std::vector<Body> bodies(NUM_BODIES);
    for (int i = 0; i < NUM_BODIES; i++) {
        bodies[i].x = rand() % WINDOW_SIZE;
        bodies[i].y = rand() % WINDOW_SIZE;
        bodies[i].vx = ((rand() % 100) - 50) / 500.0;
        bodies[i].vy = ((rand() % 100) - 50) / 500.0;
        bodies[i].mass = (rand() % 90) + 10;
        bodies[i].color = sf::Color(rand() % 255, rand() % 255, rand() % 255);
    }

    bool running = true;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    int clickedIndex = find_clicked_body(bodies, event.mouseButton.x, event.mouseButton.y);
                    if (clickedIndex != -1) {
                        bodies[clickedIndex].mass += MASS_INCREASE;
                    } else {
                        Body newBody;
                        newBody.x = event.mouseButton.x;
                        newBody.y = event.mouseButton.y;
                        newBody.vx = ((rand() % 100) - 50) / 500.0;
                        newBody.vy = ((rand() % 100) - 50) / 500.0;
                        newBody.mass = (rand() % 100) + 10;
                        newBody.color = sf::Color(rand() % 255, rand() % 255, rand() % 255);
                        bodies.push_back(newBody);
                    }
                }
            }

            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Space) running = !running;
                if (event.key.code == sf::Keyboard::R) bodies.clear();
            }
        }

        if (running && !bodies.empty()) {
            compute_forces(bodies);
            update_positions(bodies);
            handle_collisions(bodies);
        }

        window.clear();

    
        for (const auto& body : bodies) {
            sf::CircleShape circle(std::max(2.0, 4 * body.mass / 100.0)); 
            circle.setPosition(body.x - circle.getRadius(), body.y - circle.getRadius()); 
            circle.setFillColor(body.color);
            window.draw(circle);
        }

        window.display();
    }

    return 0;
}
