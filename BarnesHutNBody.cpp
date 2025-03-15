#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <algorithm>

#define G 6.67430e-11
#define TIMESTEP 1.0
#define WINDOW_SIZE 800
#define NUM_BODIES 500
#define MASS_INCREASE 50
#define THREADS 8
#define COLLISION_RADIUS 10.0
#define SOFTENING_FACTOR 1e-4
#define THETA 0.5  // Barnes-Hut threshold

struct Body {
    double x, y, vx, vy, ax, ay, mass;
    sf::Color color;
};

struct Quad {
    double x, y, size;
    bool contains(double bx, double by) const {
        return bx >= x && bx <= x + size && by >= y && by <= y + size;
    }
};

struct Node {
    Quad quad;
    bool isLeaf;
    double mass, cx, cy;
    Body* body;
    Node* children[4];
    
    Node(Quad q) : quad(q), isLeaf(true), mass(0), cx(0), cy(0), body(nullptr) {
        for (int i = 0; i < 4; i++) children[i] = nullptr;
    }
    
    ~Node() { for (int i = 0; i < 4; i++) delete children[i]; }
    
    void insert(Body* b) {
        if (!quad.contains(b->x, b->y)) return;
        if (isLeaf && body == nullptr) {
            body = b;
            return;
        }
        if (isLeaf) subdivide();
        updateMass(b);
        for (int i = 0; i < 4; i++)
            if (children[i]->quad.contains(b->x, b->y)) {
                children[i]->insert(b);
                break;
            }
    }
    
    void updateMass(Body* b) {
        mass += b->mass;
        cx = (cx * (mass - b->mass) + b->x * b->mass) / mass;
        cy = (cy * (mass - b->mass) + b->y * b->mass) / mass;
    }
    
    void subdivide() {
        double hs = quad.size / 2;
        children[0] = new Node({quad.x, quad.y, hs});
        children[1] = new Node({quad.x + hs, quad.y, hs});
        children[2] = new Node({quad.x, quad.y + hs, hs});
        children[3] = new Node({quad.x + hs, quad.y + hs, hs});
        isLeaf = false;
        if (body) {
            for (int i = 0; i < 4; i++)
                if (children[i]->quad.contains(body->x, body->y)) {
                    children[i]->insert(body);
                    break;
                }
            body = nullptr;
        }
    }
};

void compute_forces(Node* tree, Body& b) {
    if (!tree || (tree->isLeaf && tree->body == &b)) return;
    double dx = tree->cx - b.x;
    double dy = tree->cy - b.y;
    double distSq = dx * dx + dy * dy + SOFTENING_FACTOR;
    double dist = std::sqrt(distSq);
    if (tree->isLeaf || (tree->quad.size / dist < THETA)) {
        double force = (G * b.mass * tree->mass) / distSq;
        b.ax += force * dx / dist;
        b.ay += force * dy / dist;
    } else {
        for (int i = 0; i < 4; i++) compute_forces(tree->children[i], b);
    }
}

void update_positions(std::vector<Body>& bodies) {
    #pragma omp parallel for schedule(dynamic) num_threads(THREADS)
    for (int i = 0; i < bodies.size(); i++) {
        bodies[i].vx += bodies[i].ax * TIMESTEP * 0.5;
        bodies[i].vy += bodies[i].ay * TIMESTEP * 0.5;
        bodies[i].x += bodies[i].vx * TIMESTEP;
        bodies[i].y += bodies[i].vy * TIMESTEP;
    }
}

int main() {
    sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Barnes-Hut N-Body Simulation");
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
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Space)
                running = !running;
        }
        if (running) {
            Quad universe = {0, 0, WINDOW_SIZE};
            Node* root = new Node(universe);
            #pragma omp parallel for schedule(dynamic) num_threads(THREADS)
            for (int i = 0; i < bodies.size(); i++) {
                #pragma omp critical
                root->insert(&bodies[i]);
            }
            #pragma omp parallel for schedule(dynamic) num_threads(THREADS)
            for (int i = 0; i < bodies.size(); i++) {
                bodies[i].ax = bodies[i].ay = 0;
                compute_forces(root, bodies[i]);
            }
            update_positions(bodies);
            delete root;
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
