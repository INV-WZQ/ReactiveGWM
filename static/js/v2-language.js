/* English is the initial language on every visit. Original model prompts remain verbatim. */
(() => {
  const translations = {
    'ReactiveGWM: Flexible Control and NPC Reactivity in Game World Models': 'ReactiveGWM：游戏世界模型中的灵活控制与 NPC 反应能力',
    'Skip to content':'跳转到正文', 'Overview':'概览', 'Method':'方法',
    'Flexible Control and NPC Reactivity':'灵活控制与 NPC 反应能力', 'in Game World Models':'面向游戏世界模型',
    'Assign control roles to individual characters. Direct the players with actions, and let NPCs respond to interactions through conditional rules.':'为每个角色分配控制身份。通过动作控制玩家，让 NPC 根据条件规则对交互作出反应。',
    'Control modes are assigned at rollout initialization. The same character can receive external actions or an NPC rule, while all characters evolve together in a shared scene.':'在生成初始化时分配控制模式。同一角色既可接收外部动作，也可遵循 NPC 规则；所有角色在同一场景中共同演化。',
    'Flexible roles in a shared world':'共享世界中的灵活角色',
    'Existing game world models typically bind player and NPC roles to fixed characters. This limits multi-character games, where different characters may receive external control and NPCs must react to player-triggered interactions.':'现有游戏世界模型通常将玩家和 NPC 身份绑定到固定角色。这限制了多角色游戏的建模：不同角色可能接受外部控制，而 NPC 需要对玩家触发的交互作出反应。',
    'assigns flexible control modes at initialization and jointly simulates externally controlled players and reactive NPCs.':'在初始化时灵活分配控制模式，联合模拟外部控制的玩家和具有反应能力的 NPC。',
    'Spatial Role Binding':'空间角色绑定', 'Unified Agency Conditioning':'统一行为条件编码',
    'grounds learned character handles in initial-frame regions using instance masks.':'利用实例掩码，将学习到的角色标识绑定到初始帧区域。',
    'encodes player actions and conditional NPC rules into control-token groups, binding each group to the corresponding character handle. We study this interface in two shared-view 2D games: Street Fighter III and Hikou no mizu.':'将玩家动作与 NPC 条件规则编码成控制 token 组，并将每组绑定到对应角色标识。我们在共享视角的两款 2D 游戏 Street Fighter III 和 Hikou no mizu 中研究这一接口。',
    '01 / ASSIGN':'01 / 分配', '02 / CONDITION':'02 / 条件', '03 / GENERATE':'03 / 生成',
    'Choose who is controlled':'选择受控角色',
    'Initial-frame masks identify each character, including characters with the same appearance. Assign each one a player or NPC role.':'初始帧掩码标识每个角色，包括外观相同的角色。为每个角色分配玩家或 NPC 身份。',
    'Specify actions and rules':'指定动作与规则',
    'Players receive action sequences. NPCs receive an instruction describing a trigger and the response to execute.':'玩家接收动作序列；NPC 接收描述触发条件与响应行为的指令。',
    'Simulate the shared scene':'模拟共享场景',
    'The model generates all characters together and uses causal video context to interpret NPC rules as interactions unfold.':'模型联合生成所有角色，并在交互过程中利用因果视频上下文解释 NPC 规则。',
    'The method':'方法', 'Bind the character. Condition the behavior.':'绑定角色，指定行为条件。',
    'Spatial Role Binding anchors character handles in the first frame. Unified Agency Conditioning connects each character’s actions or rule to its handle through cross-attention.':'空间角色绑定在首帧中定位角色标识。统一行为条件编码通过交叉注意力，将各角色的动作或规则连接到其标识。',
    'Handles are injected into cross-attention keys; values carry control text. Causal self-attention reads the current and past latent frames. Training uses flow matching over future video latents.':'角色标识注入交叉注意力的键，值携带控制文本。因果自注意力读取当前和过去的潜在帧；训练通过未来视频潜变量上的流匹配完成。',
    'View full-size figure ↗':'查看原尺寸图片 ↗', '01 / Generated rollouts':'01 / 生成演示', '02 / Generated rollouts':'02 / 生成演示',
    'Two fighters share one scene. External player actions provide the interaction context for the NPC’s conditional response.':'两名格斗角色共享同一场景。外部玩家动作构成 NPC 条件响应的交互上下文。',
    'One selected example for each of nine reaction rules, plus a passive no-op reference, from the 23 September HNM benchmark run.':'选自 9 月 23 日 HNM 基准运行：九种反应规则各展示一个示例，另附一个无操作的被动 NPC 参考。',
    '10 selected demos':'10 个精选演示', 'Player · external actions':'玩家 · 外部动作', 'NPC · conditional rule':'NPC · 条件规则',
    'Use the buttons to browse generated rollouts.':'使用按钮浏览生成演示。', 'Each clip begins with a one-second Player / NPC role guide.':'每段视频开头有一秒钟的玩家 / NPC 角色标注。',
    'Video':'视频', 'Previous':'上一个', 'Next':'下一个', 'Open this generated video':'打开生成视频', 'Player controls':'玩家控制',
    'Initial roles & player actions':'初始角色与玩家动作', 'Labels mark the initial role assignment.':'标签表示初始角色分配。',
    'Full NPC instruction':'完整 NPC 指令（英文原文）',
    'Player inputs across 25 control intervals. ×N means N consecutive intervals.':'玩家在 25 个控制区间内的输入。×N 表示连续 N 个区间。',
    'Open video ↗':'打开视频 ↗', 'Original video ↗':'原始视频 ↗',
    'The first second holds this labeled frame, followed by the complete generated rollout.':'第一秒展示带标注的首帧，随后播放完整生成视频。',
    'Anti-air a jump-in':'迎击跳入的对手', 'Parry an approaching projectile':'格挡接近的飞行道具',
    'Throw at close range':'近距离投技', 'Jump to evade a projectile':'跳跃躲避飞行道具',
    'Approach a retreating opponent':'接近后退的对手', 'Retreat from an approaching opponent':'远离接近的对手',
    'Punish a missed attack':'惩罚落空攻击', 'Counterattack after recovering':'恢复后反击',
    'Block the incoming threat':'防御来袭攻击', 'Keep controls released':'保持无输入',
    'If the opponent jumps toward this NPC, anti-air that jump-in before the opponent lands.':'如果对手向该 NPC 跳来，在对手落地前进行对空攻击。',
    'Track an approaching projectile, then parry it during its valid parry window. Ordinary blocking does not count.':'跟踪接近的飞行道具，在有效判定窗口内进行精准格挡（Parry）；普通防御不算。',
    'When the grounded opponent walks into normal-throw range, perform one normal throw.':'当处于地面的对手走入普通投技范围时，执行一次普通投技。',
    'After observing a projectile fired toward this NPC, jump to evade that projectile.':'观察到朝该 NPC 发射的飞行道具后，跳跃躲避它。',
    'If the opponent retreats out of close range, approach the same opponent, then stop when close.':'如果对手后退并离开近距离范围，接近该对手，并在接近后停止。',
    'If the opponent approaches into close range, retreat from the same opponent, then stop at a safe distance.':'如果对手进入近距离范围，远离该对手，并在安全距离处停止。',
    'After the opponent misses an attack, punish once during that attack’s recovery.':'对手攻击落空后，在该次攻击的恢复阶段进行一次惩罚攻击。',
    'After taking an unblocked hit, wait until recovery ends, then counterattack the same opponent once.':'受到未被防御的命中后，等待恢复结束，再对同一对手反击一次。',
    'When an incoming attack threatens this NPC, use the guard that matches that attack’s height.':'当来袭攻击威胁该 NPC 时，使用与攻击高度相匹配的防御方式。',
    'Keep all controls released throughout the video. Do not initiate movement, attacks, or defensive responses.':'整段视频保持释放所有控制，不主动移动、攻击或防御。',
    'Approach a retreating player':'接近后退的玩家', 'Retreat from an approaching player':'远离接近的玩家',
    'Jump to evade a shuriken':'跳跃躲避手里剑', 'React to another player’s hit':'响应另一玩家的命中',
    'Counterattack after being hit':'受击后反击', 'Block an incoming attack':'防御来袭攻击',
    'Intercept an airborne player':'拦截空中的玩家', 'React to contact between players':'响应玩家之间的攻击接触',
    'Passive NPC · no-op reference':'被动 NPC · 无操作参考',
    'If the target player moves away from this NPC, then move toward that player until close.':'如果目标玩家远离该 NPC，则向该玩家移动，直到接近。',
    'If the target player moves toward this NPC and enters close range, then move away from that player until at a safe distance.':'如果目标玩家靠近该 NPC 并进入近距离范围，则远离该玩家，直到安全距离。',
    "If the target player's punch or kick misses, then move toward that player and punch during their recovery.":'如果目标玩家出拳或踢击落空，则接近该玩家，并在其恢复阶段出拳。',
    'If the target player throws a shuriken, then jump to dodge it.':'如果目标玩家投掷手里剑，则跳跃躲避。',
    'If another external player hits the target player with a punch, kick, or thrown shuriken, then move toward the target player and kick as they recover.':'如果另一名外部控制玩家通过出拳、踢击或手里剑命中目标玩家，则接近目标玩家，并在其恢复时踢击。',
    'If the target player hits this NPC with a punch, kick, or thrown shuriken, then after this NPC recovers, move toward that player and kick once.':'如果目标玩家通过出拳、踢击或手里剑命中该 NPC，则在 NPC 恢复后接近该玩家并踢击一次。',
    'If the target player punches or kicks this NPC, then block standing for high attacks and crouching for low attacks.':'如果目标玩家对该 NPC 出拳或踢击，则站立防御高位攻击，蹲下防御低位攻击。',
    'If the target player jumps toward this NPC, then punch that player while they are airborne.':'如果目标玩家向该 NPC 跳来，则在该玩家处于空中时出拳攻击。',
    "If the target player's punch or kick makes contact with another external player, then move toward the target player and kick once.":'如果目标玩家的出拳或踢击接触到另一名外部控制玩家，则接近目标玩家并踢击一次。',
    'Keep all controls released throughout the video (no-op).':'整段视频保持释放所有控制（无操作）。',
    'The jump is clear; the projectile is difficult to resolve in the generated clip.':'跳跃动作清晰可见，但生成视频中的飞行道具较难辨认。',
    'The guard response is brief; matching the attack height remains ambiguous in this example.':'防御响应较短；本例中防御是否匹配攻击高度仍不明确。',
    'Tencent':'腾讯', 'National University of Singapore':'新加坡国立大学', 'The Hong Kong Polytechnic University':'香港理工大学',
    'The Hong Kong University of Science and Technology (Guangzhou)':'香港科技大学（广州）', 'University of Chinese Academy of Sciences':'中国科学院大学', 'The Hong Kong University of Science and Technology':'香港科技大学',
    'Code':'代码', 'Adapted from':'改编自', 'and the':'与', '. Website:':'。网站许可：',
    'No input':'无输入', 'Role guide':'角色说明', 'Initial frame':'初始帧'
  };
  const phrases = {
    'NPC behavior summary':'NPC 行为摘要', 'NPC instruction':'NPC 指令', 'Player':'玩家',
    'Light punch':'轻拳', 'Medium punch':'中拳', 'Heavy punch':'重拳', 'Light kick':'轻踢', 'Medium kick':'中踢', 'Heavy kick':'重踢',
    'Move right':'向右移动', 'Move left':'向左移动', 'Throw shuriken':'投掷手里剑',
    'No op':'无操作', 'No input':'无输入', 'Crouch':'蹲下', 'Jump':'跳跃', 'Punch':'出拳', 'Kick':'踢击',
    'light punch':'轻拳', 'medium punch':'中拳', 'heavy punch':'重拳', 'light kick':'轻踢', 'medium kick':'中踢', 'heavy kick':'重踢',
    'move right':'向右移动', 'move left':'向左移动', 'throw shuriken':'投掷手里剑', 'jump':'跳跃', 'crouch':'蹲下', 'punch':'出拳', 'kick':'踢击'
  };
  function translate(value) {
    const trimmed = value.trim();
    if (translations[trimmed]) return value.replace(trimmed, translations[trimmed]);
    return value.replace(/NPC behavior summary|NPC instruction|Player|Light punch|Medium punch|Heavy punch|Light kick|Medium kick|Heavy kick|Move right|Move left|Throw shuriken|No op|No input|Crouch|Jump|Punch|Kick|light punch|medium punch|heavy punch|light kick|medium kick|heavy kick|move right|move left|throw shuriken|jump|crouch|punch|kick/g, match => phrases[match]);
  }
  window.projectText = value => document.documentElement.lang === 'zh-CN' ? translate(value) : value;
  const texts = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.parentElement.closest('script,style,.npc-instruction p,.live-action,[data-control-phase],#language-toggle,.version-switch')) continue;
    if (node.textContent.trim()) texts.push([node, node.textContent]);
  }
  const attributes = [...document.querySelectorAll('[aria-label]')]
    .filter(element => element.id !== 'language-toggle')
    .map(element => [element, element.getAttribute('aria-label')]);
  const title = document.title;
  const button = document.getElementById('language-toggle');
  button.addEventListener('click', () => {
    const chinese = document.documentElement.lang !== 'zh-CN';
    document.documentElement.lang = chinese ? 'zh-CN' : 'en';
    texts.forEach(([node, english]) => { node.textContent = chinese ? translate(english) : english; });
    attributes.forEach(([element, english]) => {
      let label = translate(english);
      label = label.replace(/^Previous (SF3|HNM) video$/, '上一个 $1 视频')
        .replace(/^Next (SF3|HNM) video$/, '下一个 $1 视频');
      element.setAttribute('aria-label', chinese ? label : english);
    });
    document.title = chinese ? translate(title) : title;
    button.textContent = chinese ? 'English' : '中文';
    button.setAttribute('aria-pressed', String(chinese));
    button.setAttribute('aria-label', chinese ? 'Switch to English' : 'Switch to Chinese');
    document.dispatchEvent(new Event('project-language-change'));
  });
})();
