// Footer year
document.getElementById('year').textContent = new Date().getFullYear();

// Typewriter effect
const roles = [
  "AI / ML Engineer",
  "Data Scientist",
  "Deep Learning Researcher",
  "Lecturer & Mentor"
];
const twEl = document.getElementById('typewriter-text');
let roleIndex = 0, charIndex = 0, deleting = false;

function typeLoop(){
  const current = roles[roleIndex];
  if(!deleting){
    twEl.textContent = current.slice(0, charIndex + 1);
    charIndex++;
    if(charIndex === current.length){ deleting = true; setTimeout(typeLoop, 1400); return; }
  } else {
    twEl.textContent = current.slice(0, charIndex - 1);
    charIndex--;
    if(charIndex === 0){ deleting = false; roleIndex = (roleIndex + 1) % roles.length; }
  }
  setTimeout(typeLoop, deleting ? 45 : 85);
}
typeLoop();

// Mobile nav toggle
const hamburger = document.getElementById('hamburger');
const navLinks = document.getElementById('navLinks');
hamburger.addEventListener('click', () => navLinks.classList.toggle('open'));
navLinks.querySelectorAll('a').forEach(a => a.addEventListener('click', () => navLinks.classList.remove('open')));

// Active nav link on scroll
const sections = document.querySelectorAll('section[id]');
const navAnchors = document.querySelectorAll('.nav-links a');
window.addEventListener('scroll', () => {
  let current = '';
  sections.forEach(sec => {
    const top = sec.offsetTop - 120;
    if(window.scrollY >= top) current = sec.id;
  });
  navAnchors.forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === '#' + current);
  });
});

// Contact form (front-end only demo)
const form = document.getElementById('send-message-form');
const toast = document.getElementById('toast');
form.addEventListener('submit', (e) => {
  e.preventDefault();
  toast.classList.add('show');
  form.reset();
  setTimeout(() => toast.classList.remove('show'), 3000);
});
