//
//  ViewController.swift
//  Instagram
//
//  Created by Arjun Trivedi on 2/23/21.
//

import UIKit

class ViewController: UIViewController {

    
    //@IBOutlet var imageView: UIImageView!
    @IBOutlet weak var categoryLabel: UILabel!
    
    //Created model as an object
    let model = PlantDiseaseClassifier_1()

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
    
    //this method is EXTREMELY CRUCIAL
    //It allows me to add a reference from thhe tap gesture recognizer
    // to the "ImageView" itself. Without it, this wouldn't connect properly.
    @IBOutlet private var imageView: UIImageView! {
        didSet{
            imageView.isUserInteractionEnabled = true
            imageView.image = UIImage(named: "landscape")
        }
    }
    @IBAction func imageTapped(_ sender: UITapGestureRecognizer) {
        let imageView = sender.view as? UIImageView
        
        if let imageToAnalyse = imageView?.image {
            if let sceneLabelString = sceneLabel(forImage: imageToAnalyse) {
                categoryLabel.text = sceneLabelString
            }
        }
    }
    func sceneLabel (forImage image:UIImage) -> String? {
        if let pixelBuffer = ImageProcessor.pixelBuffer(forImage: image.cgImage!) {
            guard let scene = try? model.prediction(image: pixelBuffer) else {fatalError("Unexpected runtime error")}
            return scene.classLabel
            
        }
        
        return nil
    }

    @IBAction func didTapButton(){
        let vc = UIImagePickerController()
        vc.sourceType = .photoLibrary
        vc.delegate = self
        vc.allowsEditing = true
        present(vc, animated: true)
        
    }

}


extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
                
        if let image = info[UIImagePickerController.InfoKey(rawValue: "UIImagePickerControllerEditedImage")] as? UIImage{
            imageView.image = image

        }
        
        picker.dismiss(animated: true, completion: nil)
        
    }
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
}
