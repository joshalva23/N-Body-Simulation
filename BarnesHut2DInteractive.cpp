#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <algorithm>

#define G 6.67430e-11
#define TIMESTEP 1.0
#define WINDOW_SIZE 800
#define NUM_BODIES 200
#define MASS_INCREASE 50
#define THREADS 8
#define COLLISION_RADIUS 10.0
#define THETA 0.5  // Barnes-Hut threshold

struct Body {
    double x, y, vx, vy, ax, ay, mass;
    sf::Color color;
};

struct QuadTreeNode {
    double x, y, size, mass, cm_x, cm_y;
    Body* body;
    QuadTreeNode* children[4];
    
    QuadTreeNode(double x_, double y_, double size_) : x(x_), y(y_), size(size_), mass(0), cm_x(0), cm_y(0), body(nullptr) {
        for (auto &child : children) child = nullptr;
    }
    
    ~QuadTreeNode() {
        for (auto &child : children) delete child;
    }
    
    bool contains(Body* b) {
        return (b->x >= x && b->x < x + size && b->y >= y && b->y < y + size);
    }
    
    void insert(Body* b);
    void computeMassDistribution();
    void applyForces(Body& b);
};

void QuadTreeNode::insert(Body* b) {
    if (!contains(b)) return;

    if (!body && mass == 0) {  // First body
        body = b;
        mass = b->mass;
        cm_x = b->x;
        cm_y = b->y;
        return;
    }

    if (!body) {  // Update mass and center of mass
        mass += b->mass;
        cm_x = (cm_x * (mass - b->mass) + b->x * b->mass) / mass;
        cm_y = (cm_y * (mass - b->mass) + b->y * b->mass) / mass;
    }

    if (body) {  // If already contains a body, subdivide
        Body* oldBody = body;
        body = nullptr;
        insert(oldBody);
    }

    int index = (b->x >= x + size / 2) + 2 * (b->y >= y + size / 2);
    if (!children[index]) {
        children[index] = new QuadTreeNode(
            x + (index % 2) * size / 2, 
            y + (index / 2) * size / 2, 
            size / 2
        );
    }
    children[index]->insert(b);
}


void QuadTreeNode::computeMassDistribution() {
    if (body) return;
    mass = cm_x = cm_y = 0;
    for (auto &child : children) {
        if (child) {
            child->computeMassDistribution();
            mass += child->mass;
            cm_x += child->cm_x * child->mass;
            cm_y += child->cm_y * child->mass;
        }
    }
    if (mass > 0) {
        cm_x /= mass;
        cm_y /= mass;
    }
}

void QuadTreeNode::applyForces(Body& b) {
    if (!mass || &b == body) return;
    double dx = cm_x - b.x, dy = cm_y - b.y;
    double distSq = dx * dx + dy * dy;
    double dist = sqrt(distSq);
    
    if (size / dist < THETA || body) {
        double force = (G * b.mass * mass) / (distSq + 1e-4);
        b.ax += force * dx / dist / b.mass;
        b.ay += force * dy / dist / b.mass;
    } else {
        for (auto &child : children) {
            if (child) child->applyForces(b);
        }
    }
}

void update_positions(std::vector<Body>& bodies) {
    #pragma omp parallel for schedule(dynamic) num_threads(THREADS)
    for (auto &body : bodies) {
        body.vx += body.ax * TIMESTEP * 0.5;
        body.vy += body.ay * TIMESTEP * 0.5;
        body.x += body.vx * TIMESTEP;
        body.y += body.vy * TIMESTEP;
    }
}

void handle_collisions(std::vector<Body>& bodies) {
    std::vector<bool> toRemove(bodies.size(), false);

    #pragma omp parallel for schedule(dynamic) num_threads(THREADS)
    for (size_t i = 0; i < bodies.size(); i++) {
        if (toRemove[i]) continue;
        for (size_t j = i + 1; j < bodies.size(); j++) {
            if (toRemove[j]) continue;
            double dx = bodies[j].x - bodies[i].x, dy = bodies[j].y - bodies[i].y;
            if (dx * dx + dy * dy < COLLISION_RADIUS * COLLISION_RADIUS) {
                bodies[i].vx = (bodies[i].vx * bodies[i].mass + bodies[j].vx * bodies[j].mass) / (bodies[i].mass + bodies[j].mass);
                bodies[i].vy = (bodies[i].vy * bodies[i].mass + bodies[j].vy * bodies[j].mass) / (bodies[i].mass + bodies[j].mass);
                bodies[i].mass += bodies[j].mass;
                toRemove[j] = true;
            }
        }
    }

    // Safe single-threaded pass
    bodies.erase(std::remove_if(bodies.begin(), bodies.end(), [&](const Body& b) {
        return toRemove[&b - &bodies[0]];
    }), bodies.end());
}


int main() {
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Barnes-Hut N-Body Simulation");
    std::vector<Body> bodies(NUM_BODIES);
    for (auto &body : bodies) {
        body.x = std::min(std::max(10.0, (double)(rand() % WINDOW_SIZE)), WINDOW_SIZE - 10.0);
        body.y = std::min(std::max(10.0, (double)(rand() % WINDOW_SIZE)), WINDOW_SIZE - 10.0);
        body.vx = ((rand() % 100) - 50) / 500.0;
        body.vy = ((rand() % 100) - 50) / 500.0;
        body.mass = (rand() % 90) + 10;
        body.color = sf::Color(rand() % 255, rand() % 255, rand() % 255);
    }
    
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
        }
        
        QuadTreeNode root(0, 0, WINDOW_SIZE);
        for (auto &body : bodies) root.insert(&body);
        root.computeMassDistribution();
        #pragma omp parallel for schedule(dynamic) num_threads(THREADS)
        for (auto &body : bodies) root.applyForces(body);
        
        update_positions(bodies);
        handle_collisions(bodies);
        
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
