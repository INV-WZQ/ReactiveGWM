(function () {
  const STORAGE_KEY = 'reactivegwm-language';
  const CHINESE = 'zh-CN';
  const ENGLISH = 'en';

  const textTranslations = new Map([
    ['ReactiveGWM: Steering NPC in Reactive Game World Models', 'ReactiveGWM：面向反应式游戏世界模型的 NPC 策略操控'],
    ['Tencent', '腾讯'],
    ['National University of Singapore', '新加坡国立大学'],
    ['The Hong Kong Polytechnic University', '香港理工大学'],
    ['The Hong Kong University of Science and Technology (Guangzhou)', '香港科技大学（广州）'],
    ['University of Chinese Academy of Sciences', '中国科学院大学'],
    ['The Hong Kong University of Science and Technology', '香港科技大学'],
    ['Code', '代码'],
    ['Method', '方法'],
    ['Configurations', '配置'],
    ['A game world model where the', '一种让'],
    ['NPC follows high-level strategies', 'NPC 遵循高层策略'],
    [', not just appears as background pixels — and the strategy module', '、而非仅作为背景像素出现的游戏世界模型；其策略模块还可'],
    ['transfers zero-shot', '零样本迁移'],
    ['to a new game.', '到新游戏。'],
    ['ReactiveGWM decouples player control from NPC behavior: player buttons enter the diffusion backbone as a lightweight additive bias, while NPC intents (Offense / Defense / Control) are grounded through cross-attention. Trained on one game, the cross-attention modules plug directly into an unannotated world model of a different game — unlocking steerable NPCs without retraining.', 'ReactiveGWM 将玩家控制与 NPC 行为解耦：玩家按键以轻量加性偏置的形式进入扩散主干，而 NPC 意图（进攻 / 防守 / 控制）则通过交叉注意力进行条件控制。在一个游戏上训练后，交叉注意力模块可直接接入另一个游戏未经标注的世界模型，无需重新训练即可实现可操控的 NPC。'],
    ['Decoupling player control and NPC strategy', '解耦玩家控制与 NPC 策略'],
    ['Two non-interfering pathways inside the DiT block: an', 'DiT 模块内包含两条互不干扰的路径：用于细粒度玩家按键的'],
    ['additive bias', '加性偏置'],
    ['for fine-grained player buttons, and', '，以及用于高层 NPC 策略的'],
    ['cross-attention', '交叉注意力'],
    ["for high-level NPC strategy. Self-attention and FFN keep modeling the game's native dynamics.", '。自注意力和 FFN 继续建模游戏自身的动态。'],
    ['Strategy-aligned data', '策略对齐数据'],
    ['Each clip is paired with an', '每个视频片段都配有一条'],
    ['NPC-only structured prompt', '仅描述 NPC 的结构化提示词'],
    ['— a strategy (Offense / Defense / Control) plus active & passive behaviors — separated from player actions and scene captions.', '——包含一种策略（进攻 / 防守 / 控制）以及主动与被动行为，并与玩家动作和场景描述分离。'],
    ['Construction of strategy-aligned data: every clip is annotated with player actions and an NPC-only structured prompt.', '策略对齐数据的构建：每个片段均标注玩家动作和仅描述 NPC 的结构化提示词。'],
    ['Two pathways inside the DiT block', 'DiT 模块内的两条路径'],
    ['Player buttons', '玩家按键'],
    ['are pooled to the latent frame rate and added as a residual bias.', '先聚合到潜在帧率，再作为残差偏置加入。'],
    ['NPC strategy', 'NPC 策略'],
    ['is encoded as text and injected via cross-attention. Self-attention and FFN are left intact.', '被编码为文本，并通过交叉注意力注入；自注意力与 FFN 保持不变。'],
    ['DiT block with action module (additive bias) and strategy cross-attention.', '带有动作模块（加性偏置）和策略交叉注意力的 DiT 模块。'],
    ['Train once, transfer zero-shot', '一次训练，零样本迁移'],
    ['base', '基础模型'],
    [': full fine-tuning on a source game with strategy annotations.', '：在带有策略标注的源游戏上进行完整微调。'],
    ['transfer', '迁移模型'],
    [": reuse the target game's vanilla backbone and plug in our trained cross-attention — steerable NPCs without any retraining on the new game.", '：复用目标游戏的原始主干，并接入训练好的交叉注意力模块；无需在新游戏上重新训练，即可实现可操控的 NPC。'],
    ['Overview of ReactiveGWM training and training-free transfer to a different game.', 'ReactiveGWM 的训练流程，以及无需训练即可迁移到不同游戏的概览。'],
    ['Control interface', '控制接口'],
    ['The', '其中'],
    ['player', '玩家'],
    ['is controlled via low-level', '由低层'],
    ['buttons', '按键'],
    ['; the', '控制；'],
    ['NPC', 'NPC'],
    ['is steered via a high-level', '则由高层'],
    ['strategy', '策略'],
    ['.', '进行操控。'],
    ['Player — controlled by buttons', '玩家——通过按键控制'],
    ['NPC — steered by strategy', 'NPC——通过策略操控'],
    ['Demo · Street Fighter 2', '演示 · 街头霸王 2'],
    ['Same buttons, different strategies', '相同按键，不同策略'],
    ['Compare the vanilla backbone,', '在相同玩家输入、不同 NPC 策略（进攻 / 防守 / 控制）下，对比原始主干、'],
    [', and', '、'],
    ['under the same player input but different NPC strategies (Offense / Defense / Control).', '。'],
    ['SF2 Button Mapping', 'SF2 按键映射'],
    ['Light Punch (LP)', '轻拳（LP）'],
    ['Medium Punch (MP)', '中拳（MP）'],
    ['Heavy Punch (HP)', '重拳（HP）'],
    ['Light Kick (LK)', '轻脚（LK）'],
    ['Medium Kick (MK)', '中脚（MK）'],
    ['Heavy Kick (HK)', '重脚（HK）'],
    ['Vanilla', '原始模型'],
    ['NPC Strategy: Offense', 'NPC 策略：进攻'],
    ['NPC Strategy: Defense', 'NPC 策略：防守'],
    ['NPC Strategy: Control', 'NPC 策略：控制'],
    ['Prompt Detail', '提示词详情'],
    ['Copy Prompt', '复制提示词'],
    ['Demo · Street Fighter 3', '演示 · 街头霸王 3'],
    ['Cross-game strategy transfer', '跨游戏策略迁移'],
    ['reuses the strategy modules trained on SF2 on top of an unannotated SF3 backbone — steerable NPCs emerge without any retraining on this game.', '复用在 SF2 上训练的策略模块，并将其接入未经标注的 SF3 主干；无需在该游戏上重新训练，即可获得可操控的 NPC。'],
    ['SF3 Button Mapping', 'SF3 按键映射']
  ]);

  const attributeTranslations = new Map([
    ['Section navigation', '页面章节导航'],
    ['Demo teaser', '演示视频'],
    ['HuggingFace Model', 'HuggingFace 模型'],
    ['HuggingFace Dataset', 'HuggingFace 数据集'],
    ['Data construction pipeline: gameplay clips paired with NPC-only structured prompts', '数据构建流程：将游戏视频片段与仅描述 NPC 的结构化提示词配对'],
    ['DiT block with player additive bias and NPC strategy cross-attention', '带有玩家加性偏置和 NPC 策略交叉注意力的 DiT 模块'],
    ['Training and training-free transfer of the strategy module across games', '策略模块的训练及跨游戏免训练迁移'],
    ['Control role overview', '控制角色概览'],
    ['Frame from Street Fighter showing both characters with the player highlighted in blue and the NPC highlighted in red', '街头霸王画面：玩家以蓝色框标出，NPC 以红色框标出'],
    ['SF2 button mapping', 'SF2 按键映射'],
    ['SF3 button mapping', 'SF3 按键映射']
  ]);

  const messages = {
    en: {
      clickPrompt: '💡 Click the corresponding video to view its prompt details.',
      noVideoSelected: 'No video selected yet.',
      loadingPrompt: 'Loading prompt...',
      promptEmpty: 'Prompt is empty.',
      copyPrompt: 'Copy Prompt',
      copied: 'Copied',
      copyFailed: 'Copy Failed'
    },
    'zh-CN': {
      clickPrompt: '💡 点击对应视频可查看提示词详情。',
      noVideoSelected: '尚未选择视频。',
      loadingPrompt: '正在加载提示词……',
      promptEmpty: '提示词为空。',
      copyPrompt: '复制提示词',
      copied: '已复制',
      copyFailed: '复制失败'
    }
  };

  const promptPhraseTranslations = new Map([
    ['A kick or sweep thrown from a crouching posture (includes low sweeps).', '从蹲姿发起踢击或扫堂腿（包括下段扫腿）。'],
    ['A punch or kick thrown while airborne (jump-in or air-to-air).', '在空中发起拳击或踢击（跳入攻击或空对空攻击）。'],
    ['A scripted chain of normals into a specific follow-up unique to SF3.', '由普通攻击衔接 SF3 特有后续招式的固定连段。'],
    ['A standard foot strike (LK/MK/HK) thrown while standing.', '站立时使出的标准踢击（LK/MK/HK）。'],
    ['A standard hand strike (LP/MP/HP) thrown while standing.', '站立时使出的标准拳击（LP/MP/HP）。'],
    ['A two-button enhanced version of any special, marked by a blue flash and extra hits/properties.', '任意必杀技的双键强化版本，会出现蓝色闪光并增加命中次数或特殊属性。'],
    ['Absorbs a hit and enters hitstun with health loss.', '承受一次攻击，损失生命值并进入受击硬直。'],
    ["Absorbs a hit from an opponent's attack, resulting in hitstun and health loss.", '承受对手攻击，损失生命值并进入受击硬直。'],
    ['Absorbs and evades incoming pressure, recovering safely instead of trading hits.', '承受并躲避对手压制，以安全恢复代替互相换血。'],
    ['Balances offense and defense by controlling distance, neither rushing in nor purely turtling.', '通过控制距离平衡进攻与防守，既不贸然贴身，也不一味龟缩。'],
    ['Closes the distance quickly to apply pressure and initiate close combat.', '快速拉近距离施加压力并发起近身战。'],
    ["Enters and holds a crouching stance to lower the character's hitbox and prepare charged moves.", '进入并保持蹲姿，以降低角色受击区域并准备蓄力招式。'],
    ['Executes a basic kick attack while in a standing posture.', '在站立姿态下使出基础踢击。'],
    ['Executes a basic punch attack while in a standing posture.', '在站立姿态下使出基础拳击。'],
    ['Executes a close-range grappling maneuver to toss the opponent.', '使出近距离投技，将对手抛出。'],
    ['Executes a low sweep or kick attack from a crouching posture.', '从蹲姿使出下段扫腿或踢击。'],
    ['Executes a neutral vertical jump into the air without horizontal movement.', '垂直原地起跳，不产生水平位移。'],
    ['Executes a punch attack while maintaining a crouching posture.', '保持蹲姿并使出拳击。'],
    ['Executes a punch or kick attack while airborne.', '在空中使出拳击或踢击。'],
    ['Executes an aerial jump towards the left side of the screen.', '向屏幕左侧跳跃。'],
    ['Executes an aerial jump towards the right side of the screen.', '向屏幕右侧跳跃。'],
    ['Falls to the ground after a heavy hit or sweep.', '遭受重击或扫堂腿后倒地。'],
    ['Falls to the ground after receiving a heavy or sweeping attack.', '遭受重击或扫击后倒地。'],
    ['Focuses on advancing and chaining attacks to keep the opponent on the back foot.', '专注于推进和连续攻击，使对手持续处于被动。'],
    ['Frozen briefly while blocking an attack — still safe but locked into guard.', '格挡攻击时短暂僵直；虽处于安全状态，但暂时只能保持防御。'],
    ["Guile's signature projectile attack thrown horizontally across the screen.", '古烈的标志性飞行道具，沿屏幕水平方向发出。'],
    ["Guile's signature somersault kick used primarily as an anti-air or defensive reversal.", '古烈的标志性翻身踢，主要用于对空或防守反击。'],
    ['Holds ground with blocks and reactive counters, only striking when an opening appears.', '通过格挡与反应式反击守住阵地，只在对手露出破绽时出手。'],
    ["Ibuki's rolling multi-hit kick string used for pressure and combo extension.", '伊吹用于压制和延长连段的滚动多段踢击。'],
    ["Ibuki's signature throwing-knife projectile, often performed in mid-air at a downward angle.", '伊吹的标志性苦无飞行道具，常在空中以向下角度投掷。'],
    ['Jumping in an arc toward the left side of the screen.', '沿弧线向屏幕左侧跳跃。'],
    ['Jumping in an arc toward the right side of the screen.', '沿弧线向屏幕右侧跳跃。'],
    ['Jumping straight vertically without lateral movement.', '垂直向上跳跃，不产生横向位移。'],
    ['Lowering the body into a squatting stance to avoid high attacks or prepare low attacks.', '降低身体进入蹲姿，以躲避上段攻击或准备下段攻击。'],
    ['Maintains constant aggression to overwhelm the opponent and force defensive reactions.', '保持持续进攻以压制对手，迫使其采取防守反应。'],
    ['Manages spacing with projectiles and measured pokes to dictate the pace of engagement.', '利用飞行道具和谨慎的试探攻击管理距离，掌控交战节奏。'],
    ['Moves character towards the left side of the screen.', '使角色向屏幕左侧移动。'],
    ['Moves character towards the right side of the screen.', '使角色向屏幕右侧移动。'],
    ['Passively avoids an incoming attack or projectile.', '被动躲避来袭的攻击或飞行道具。'],
    ['Passively guards against high or mid attacks while standing upright.', '保持站立姿态，被动防御上段或中段攻击。'],
    ['Passively guards against high or mid attacks while standing.', '站立时被动防御上段或中段攻击。'],
    ['Passively guards against low and mid attacks from a lowered stance.', '以蹲姿被动防御下段或中段攻击。'],
    ["Prioritizes guarding and reading the opponent's actions over initiating offense.", '优先进行防守并判断对手行动，而非主动进攻。'],
    ["Pushed away by the force of an opponent's blocked or landed attack.", '被对手命中或被格挡攻击的冲击力推开。'],
    ['Pushed away horizontally by the force of a blocked or landed attack.', '被命中或被格挡攻击的冲击力水平推开。'],
    ['Quickly stepping or hopping toward the left side of the screen (double-tap LEFT).', '快速向屏幕左侧迈步或小跳（双击左方向）。'],
    ['Quickly stepping or hopping toward the right side of the screen (double-tap RIGHT).', '快速向屏幕右侧迈步或小跳（双击右方向）。'],
    ['Recovery animation while standing back up after a knockdown.', '倒地后重新起身时的恢复动作。'],
    ['Sidesteps or otherwise avoids an incoming attack/projectile without blocking.', '不进行格挡，通过侧移或其他方式躲避来袭的攻击或飞行道具。'],
    ['Stands perfectly still without any input, waiting for an interaction.', '不进行任何输入并完全静止，等待交互。'],
    ['Stands still without input, waiting for an interaction.', '不进行输入并保持静止，等待交互。'],
    ['The recovery animation of standing back up after being knocked down.', '被击倒后重新起身的恢复动作。'],
    ['Uses range and zoning tools to keep the opponent at a preferred distance and force reactions.', '利用射程和区域控制手段将对手维持在理想距离，并迫使其做出反应。'],
    ['Walking horizontally to the left side of the screen.', '沿水平方向向屏幕左侧移动。'],
    ['Walking horizontally to the right side of the screen.', '沿水平方向向屏幕右侧移动。']
  ]);

  const promptLabelTranslations = new Map([
    ['Active_Behavior', '主动行为'],
    ['Passive_Behavior', '被动行为'],
    ['Block Stun:', '格挡硬直：'],
    ['Crouching Block:', '蹲姿格挡：'],
    ['Crouching Kick:', '蹲姿踢击：'],
    ['Crouching Punch:', '蹲姿拳击：'],
    ['Dash Left:', '向左冲刺：'],
    ['Dash Right:', '向右冲刺：'],
    ['EX Move:', 'EX 招式：'],
    ['Flash Kick:', '闪光踢：'],
    ['Jump In Place:', '原地跳跃：'],
    ['Jump Left:', '向左跳跃：'],
    ['Jump Right:', '向右跳跃：'],
    ['Jumping Attack:', '跳跃攻击：'],
    ['Knockback:', '击退：'],
    ['Knockdown:', '击倒：'],
    ['Kunai Throw:', '苦无投掷：'],
    ['Sonic Boom:', '音速手刀：'],
    ['Standing Block:', '站立格挡：'],
    ['Standing Kick:', '站立踢击：'],
    ['Standing Punch:', '站立拳击：'],
    ['Take Damage:', '受到伤害：'],
    ['Target Combo:', '目标连段：'],
    ['Tsumuji Kicks:', '旋风踢：'],
    ['Wake Up:', '起身：'],
    ['Walk Left:', '向左行走：'],
    ['Walk Right:', '向右行走：'],
    ['Crouch:', '蹲伏：'],
    ['Evade:', '闪避：'],
    ['Idle:', '待机：'],
    ['Throw:', '投技：'],
    ['Strategy', '策略'],
    ['Offense:', '进攻：'],
    ['Defense:', '防守：'],
    ['Control:', '控制：'],
    ['NPC:', 'NPC：']
  ]);

  const strategyNames = {
    Offense: '进攻',
    Defense: '防守',
    Control: '控制'
  };

  const methodNames = {
    vanilla: '原始模型',
    base: '基础模型',
    transfer: '迁移模型'
  };

  const textRecords = [];
  const attributeRecords = [];
  let currentLanguage = ENGLISH;

  const replacePreservingWhitespace = (original, translated) => {
    const leading = original.match(/^\s*/)[0];
    const trailing = original.match(/\s*$/)[0];
    return `${leading}${translated}${trailing}`;
  };

  const collectTranslatableContent = () => {
    const dynamicSelector = '#language-toggle, .prompt-copy-btn, .prompt-detail-meta, .prompt-detail-content';
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();

    while (node) {
      const parent = node.parentElement;
      const source = node.nodeValue.trim();
      if (
        parent &&
        !parent.closest(dynamicSelector) &&
        !['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(parent.tagName) &&
        textTranslations.has(source)
      ) {
        textRecords.push({
          node,
          original: node.nodeValue,
          translated: replacePreservingWhitespace(node.nodeValue, textTranslations.get(source))
        });
      }
      node = walker.nextNode();
    }

    document.querySelectorAll('[aria-label], [alt], [title]').forEach((element) => {
      ['aria-label', 'alt', 'title'].forEach((attribute) => {
        const source = element.getAttribute(attribute);
        if (source && attributeTranslations.has(source)) {
          attributeRecords.push({
            element,
            attribute,
            original: source,
            translated: attributeTranslations.get(source)
          });
        }
      });
    });
  };

  const getMessage = (key) => messages[currentLanguage][key] || messages.en[key] || '';

  const formatSelection = (id, scenario, method) => {
    if (currentLanguage === CHINESE) {
      return `💡 已选择：${id} · ${strategyNames[scenario] || scenario} · ${methodNames[method] || method}`;
    }
    return `💡 Selected: ${id} · ${scenario} · ${method}`;
  };

  const formatPromptError = (path) => currentLanguage === CHINESE
    ? `无法从以下位置加载提示词：${path}`
    : `Failed to load prompt from: ${path}`;

  const localizePrompt = (rawPrompt) => {
    if (currentLanguage !== CHINESE) return rawPrompt;

    let translated = rawPrompt;
    [...promptPhraseTranslations.entries()]
      .sort((a, b) => b[0].length - a[0].length)
      .forEach(([source, target]) => {
        translated = translated.split(source).join(target);
      });
    promptLabelTranslations.forEach((target, source) => {
      translated = translated.split(source).join(target);
    });
    return translated
      .replaceAll('; ', '；')
      .replaceAll('), ', '），')
      .replaceAll('.)', '。）');
  };

  const refreshDynamicPanels = () => {
    ['sf2', 'sf3'].forEach((prefix) => {
      const meta = document.getElementById(`${prefix}-prompt-meta`);
      const content = document.getElementById(`${prefix}-prompt-content`);

      if (meta) {
        meta.textContent = meta.dataset.selectedId
          ? formatSelection(
              meta.dataset.selectedId,
              meta.dataset.selectedScenario,
              meta.dataset.selectedMethod
            )
          : getMessage('clickPrompt');
      }

      if (!content) return;
      const state = content.dataset.state || 'noSelection';
      if (state === 'loading') {
        content.textContent = getMessage('loadingPrompt');
      } else if (state === 'prompt') {
        content.textContent = localizePrompt(content.dataset.rawPrompt || '');
      } else if (state === 'emptyPrompt') {
        content.textContent = getMessage('promptEmpty');
      } else if (state === 'error') {
        content.textContent = formatPromptError(content.dataset.promptPath || '');
      } else {
        content.textContent = getMessage('noVideoSelected');
      }
    });
  };

  const applyLanguage = (language, persist) => {
    currentLanguage = language === CHINESE ? CHINESE : ENGLISH;
    const isChinese = currentLanguage === CHINESE;

    document.documentElement.lang = currentLanguage;
    document.title = isChinese
      ? textTranslations.get('ReactiveGWM: Steering NPC in Reactive Game World Models')
      : 'ReactiveGWM: Steering NPC in Reactive Game World Models';

    textRecords.forEach(({ node, original, translated }) => {
      node.nodeValue = isChinese ? translated : original;
    });
    attributeRecords.forEach(({ element, attribute, original, translated }) => {
      element.setAttribute(attribute, isChinese ? translated : original);
    });

    const toggle = document.getElementById('language-toggle');
    if (toggle) {
      toggle.textContent = isChinese ? 'English' : '中文';
      toggle.setAttribute('aria-label', isChinese ? '切换至英文' : 'Switch to Chinese');
      toggle.setAttribute('aria-pressed', String(isChinese));
    }

    document.querySelectorAll('.prompt-copy-btn').forEach((button) => {
      button.textContent = getMessage('copyPrompt');
    });

    refreshDynamicPanels();

    if (persist) {
      try {
        localStorage.setItem(STORAGE_KEY, currentLanguage);
      } catch (error) {
        // The language switch still works when browser storage is unavailable.
      }
    }
  };

  window.pageLanguage = {
    getLanguage: () => currentLanguage,
    getMessage,
    formatSelection,
    formatPromptError,
    localizePrompt,
    refreshDynamicPanels
  };

  document.addEventListener('DOMContentLoaded', () => {
    collectTranslatableContent();

    let initialLanguage = ENGLISH;
    try {
      initialLanguage = localStorage.getItem(STORAGE_KEY) === CHINESE ? CHINESE : ENGLISH;
    } catch (error) {
      initialLanguage = ENGLISH;
    }
    applyLanguage(initialLanguage, false);

    const toggle = document.getElementById('language-toggle');
    if (toggle) {
      toggle.addEventListener('click', () => {
        applyLanguage(currentLanguage === CHINESE ? ENGLISH : CHINESE, true);
      });
    }
  });
})();
