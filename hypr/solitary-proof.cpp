// Sealed-seat fixture: make the lone-fullscreen shortcut eligible even when
// the headless seat's diagnostic banner would normally prevent it.
#include <hyprland/src/plugins/PluginAPI.hpp>
#include <hyprland/src/desktop/state/WindowState.hpp>
#include <hyprland/src/desktop/view/Window.hpp>
#include <hyprland/src/managers/fullscreen/FullscreenController.hpp>
#include <hyprland/src/state/MonitorState.hpp>
#include <hyprland/src/output/Monitor.hpp>
#include <hyprland/src/event/EventBus.hpp>
#include <fstream>
#include <string>

static CHyprSignalListener listener;
APICALL EXPORT std::string PLUGIN_API_VERSION() { return HYPRLAND_API_VERSION; }
APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE) {
    if (std::string(__hyprland_api_get_hash()) != std::string(__hyprland_api_get_client_hash()))
        throw std::runtime_error("solitary proof: Hyprland header mismatch");
    listener = Event::bus()->m_events.render.stage.listen([](eRenderStage stage) {
        for (auto const& m : State::monitorState()->monitors()) {
            if (stage == RENDER_PRE) {
                for (auto const& w : Desktop::windowState()->windows())
                    if (w->m_isMapped && !w->isHidden() && (w->m_class == "chocolate-doom" || w->m_class == "crispy-doom") &&
                        Fullscreen::controller()->isFullscreen(w, Fullscreen::FSMODE_FULLSCREEN))
                        m->m_solitaryClient = w;
            }
            if (stage == RENDER_BEGIN)
                std::ofstream(std::string(getenv("HOME")) + "/solitary.now") << bool(m->m_solitaryClient) << "\n";
        }
    });
    return {"solitary-proof", "Fullscreen regression fixture for the sealed seat only", "Parker Brown", "1"};
}
APICALL EXPORT void PLUGIN_EXIT() { listener.reset(); }
