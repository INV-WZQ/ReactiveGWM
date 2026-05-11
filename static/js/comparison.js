// js/infoButtons.js

document.addEventListener('DOMContentLoaded', () => {
    const infoButtons = document.querySelectorAll('.info-button');
    infoButtons.forEach((button) => {
      button.addEventListener('click', function () {
        const prompt = this.dataset.prompt;
        alert(prompt);
      });
    });
  });