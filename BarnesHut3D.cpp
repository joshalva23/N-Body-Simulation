#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <random>

#define G 6.67430e-11
#define TIMESTEP 10
#define WINDOW_SIZE 800
#define NUM_BODIES 500
#define MASS_INCREASE 50
#define THREADS 8
#define COLLISION_RADIUS 5.0
#define THETA 0.5  
#define PROJECTION_DISTANCE 500  

struct Body {
    double x, y, z, vx, vy, vz, ax, ay, az, mass;
    sf::Color color;
};

struct OctreeNode {
    double x, y, z, size, mass, cm_x, cm_y, cm_z;
    Body* body;
    OctreeNode* children[8];

    OctreeNode(double x_, double y_, double z_, double size_) 
        : x(x_), y(y_), z(z_), size(size_), mass(0), cm_x(0), cm_y(0), cm_z(0), body(nullptr) {
        for (auto &child : children) child = nullptr;
    }

    ~OctreeNode() {
        for (auto &child : children) delete child;
    }

    bool contains(Body* b) {
        return (b->x >= x && b->x < x + size && 
                b->y >= y && b->y < y + size && 
                b->z >= z && b->z < z + size);
    }

    void insert(Body* b);
    void computeMassDistribution();
    void applyForces(Body& b);
};

void OctreeNode::insert(Body* b) {
    if (!contains(b)) return;

    if (!body && mass == 0) {  
        body = b;
        mass = b->mass;
        cm_x = b->x;
        cm_y = b->y;
        cm_z = b->z;
        return;
    }

    if (body) {  
        Body* oldBody = body;
        body = nullptr;  

        insert(oldBody);  
    }

    mass += b->mass;
    cm_x = (cm_x * (mass - b->mass) + b->x * b->mass) / mass;
    cm_y = (cm_y * (mass - b->mass) + b->y * b->mass) / mass;
    cm_z = (cm_z * (mass - b->mass) + b->z * b->mass) / mass;

    int index = (b->x >= x + size / 2) + 2 * (b->y >= y + size / 2) + 4 * (b->z >= z + size / 2);
    if (!children[index]) 
        children[index] = new OctreeNode(x + (index % 2) * size / 2, 
                                         y + ((index / 2) % 2) * size / 2, 
                                         z + (index / 4) * size / 2, 
                                         size / 2);
    children[index]->insert(b);
}

void OctreeNode::computeMassDistribution() {
    if (body) return;

    mass = cm_x = cm_y = cm_z = 0;
    for (auto &child : children) {
        if (child) {
            child->computeMassDistribution();
            mass += child->mass;
            cm_x += child->cm_x * child->mass;
            cm_y += child->cm_y * child->mass;
            cm_z += child->cm_z * child->mass;
        }
    }
    if (mass > 0) {
        cm_x /= mass;
        cm_y /= mass;
        cm_z /= mass;
    }
}

void OctreeNode::applyForces(Body& b) {
    if (!mass || &b == body) return;
    double dx = cm_x - b.x, dy = cm_y - b.y, dz = cm_z - b.z;
    double distSq = dx * dx + dy * dy + dz * dz;
    double dist = sqrt(distSq) + 1e-4;

    if (size / dist < THETA || body) {
        double force = (G * b.mass * mass) / (distSq);
        b.ax += force * dx / dist / b.mass;
        b.ay += force * dy / dist / b.mass;
        b.az += force * dz / dist / b.mass;
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
        body.vz += body.az * TIMESTEP * 0.5;
        body.x += body.vx * TIMESTEP;
        body.y += body.vy * TIMESTEP;
        body.z += body.vz * TIMESTEP;
    }
}

void handle_collisions(std::vector<Body>& bodies) {
    std::vector<bool> toRemove(bodies.size(), false);
    #pragma omp parallel for schedule(dynamic) num_threads(THREADS)
    for (size_t i = 0; i < bodies.size(); i++) {
        if (toRemove[i]) continue;
        for (size_t j = i + 1; j < bodies.size(); j++) {
            if (toRemove[j]) continue;
            double dx = bodies[j].x - bodies[i].x, dy = bodies[j].y - bodies[i].y, dz = bodies[j].z - bodies[i].z;
            if (dx * dx + dy * dy + dz * dz < COLLISION_RADIUS * COLLISION_RADIUS) {
                bodies[i].vx = (bodies[i].vx * bodies[i].mass + bodies[j].vx * bodies[j].mass) / (bodies[i].mass + bodies[j].mass);
                bodies[i].vy = (bodies[i].vy * bodies[i].mass + bodies[j].vy * bodies[j].mass) / (bodies[i].mass + bodies[j].mass);
                bodies[i].vz = (bodies[i].vz * bodies[i].mass + bodies[j].vz * bodies[j].mass) / (bodies[i].mass + bodies[j].mass);
                bodies[i].mass += bodies[j].mass;
                toRemove[j] = true;
            }
        }
    }
    bodies.erase(std::remove_if(bodies.begin(), bodies.end(), [&](const Body& b) { return toRemove[&b - &bodies[0]]; }), bodies.end());
}


void project3DTo2D(double x, double y, double z, double& screen_x, double& screen_y) {
    double factor = PROJECTION_DISTANCE / (PROJECTION_DISTANCE + z);
    screen_x = (x - WINDOW_SIZE / 2) * factor + WINDOW_SIZE / 2;
    screen_y = (y - WINDOW_SIZE / 2) * factor + WINDOW_SIZE / 2;
}

int main() {
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "3D Barnes-Hut N-Body Simulation");
    std::vector<Body> bodies(NUM_BODIES);
    for (auto &body : bodies) {
        body.x = rand() % WINDOW_SIZE;
        body.y = rand() % WINDOW_SIZE;
        body.z = rand() % WINDOW_SIZE;
        body.vx = ((rand() % 100) - 50) / 500.0;
        body.vy = ((rand() % 100) - 50) / 500.0;
        body.vz = ((rand() % 100) - 50) / 500.0;
        body.mass = (rand() % 90) + 10;
        body.color = sf::Color(rand() % 255, rand() % 255, rand() % 255);
    }

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) if (event.type == sf::Event::Closed) window.close();

        OctreeNode root(0, 0, 0, WINDOW_SIZE);
        for (auto &body : bodies) root.insert(&body);
        root.computeMassDistribution();

        #pragma omp parallel for num_threads(THREADS)
        for (auto &body : bodies) root.applyForces(body);

        update_positions(bodies);
        handle_collisions(bodies);

        window.clear();
        for (const auto& body : bodies) {
            double screen_x, screen_y;
            project3DTo2D(body.x, body.y, body.z, screen_x, screen_y);

            double depth_factor = std::max(0.3, 1.0 - (body.z / WINDOW_SIZE));
            sf::CircleShape circle(std::max(2.0, 6 * body.mass / 100.0 * depth_factor));
            circle.setPosition(screen_x - circle.getRadius(), screen_y - circle.getRadius());
            circle.setFillColor(sf::Color(body.color.r * depth_factor, body.color.g * depth_factor, body.color.b * depth_factor));
            window.draw(circle);
        }
        window.display();
    }
    return 0;
}
