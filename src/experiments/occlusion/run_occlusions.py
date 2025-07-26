from experiments.occlusion import gpt_roof_occlusion, llava_roof_occlusion


def run_occlusion_suite():
    tests = [
        ("Roof Occlusion Tests", gpt_roof_occlusion.main),
        ("LLaVA Occlusion Tests", llava_roof_occlusion.main),
    ]

    for name, func in tests:
        print(f"\nRunning {name}...")
        func()
