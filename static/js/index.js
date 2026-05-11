window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    // 针对custom-carousel的配置
    var customOptions = {
        slidesToScroll: 1,
        slidesToShow: 4,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 40000
    }
    // 初始化custom-carousel
    var customCarousels = bulmaCarousel.attach('.custom-carousel', customOptions)
	
    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 40000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();
	
	// ============ 新增的按钮点击处理代码 ============
    // 获取模态框相关元素
    const modal = document.getElementById('prompt-modal');
    const modalPrompt = document.getElementById('modal-prompt');
    const closeElements = [
        document.querySelector('.modal-background'),
        document.querySelector('.delete')
    ];

    // 为所有信息按钮添加点击事件
    document.querySelectorAll('.info-button').forEach(button => {
        button.addEventListener('click', function(event) {
            event.stopPropagation(); // 防止事件冒泡
            const prompt = this.getAttribute('data-prompt');
            modalPrompt.textContent = prompt;
            modal.classList.add('is-active');
        });
    });

    // 关闭模态框的公共方法
    const closeModal = () => modal.classList.remove('is-active');
    
    // 绑定关闭事件
    closeElements.forEach(element => {
        element.addEventListener('click', closeModal);
    });

    // ESC键关闭
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') closeModal();
    });
	document.querySelectorAll('.bottom-prompt-button').forEach(button => {
		button.addEventListener('click', function(event) {
		  event.stopPropagation();
		  const prompt = this.getAttribute('data-prompt');
		  modalPrompt.textContent = prompt;
		  modal.classList.add('is-active');
		});
	  });
})
